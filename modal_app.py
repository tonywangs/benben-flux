"""Modal training, inference and private Gradio UI for Benben."""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal

from benben import DIFFUSERS_COMMIT, MODEL_ID, SUBJECT, TrainConfig, prepare_photos, validate_run_id

app = modal.App("benben-flux")
volume = modal.Volume.from_name("benben-flux", create_if_missing=True)
ROOT = Path("/data")
BASE = ROOT / "base"
RUNS = ROOT / "runs"
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
web_secret = modal.Secret.from_name("benben-web-auth", required_keys=["USERNAME", "PASSWORD"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "diffusers==0.35.1",
        "transformers==4.56.2",
        "peft==0.17.1",
        "accelerate==1.10.1",
        "huggingface-hub==0.35.3",
        "sentencepiece==0.2.1",
        "protobuf==6.32.1",
        "ftfy==6.3.1",
        "tensorboard==2.20.0",
        "pillow==11.3.0",
    )
    .run_commands(
        "git init /opt/diffusers",
        "git -C /opt/diffusers remote add origin https://github.com/huggingface/diffusers",
        f"git -C /opt/diffusers fetch --depth=1 origin {DIFFUSERS_COMMIT}",
        f"git -C /opt/diffusers checkout {DIFFUSERS_COMMIT}",
    )
    .env({"HF_HOME": "/data/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("benben")
)
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("gradio==5.49.1", "fastapi==0.116.1", "huggingface-hub==0.35.3")
    .add_local_python_source("benben")
)


@app.function(
    image=image, volumes={ROOT: volume}, secrets=[hf_secret], timeout=3600, max_containers=1
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        local_dir=BASE,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "transformer/*",
            "vae/*",
        ],
        ignore_patterns=["*.bin", "*.pt", "*.msgpack"],
    )
    (BASE / "READY").write_text(MODEL_ID)
    volume.commit()


@app.function(image=image, volumes={ROOT: volume}, timeout=600, max_containers=1)
def upload_dataset(run_id: str, photos: list[bytes], config: dict):
    validate_run_id(run_id)
    target = RUNS / run_id
    if target.exists():
        raise ValueError(f"Run {run_id} already exists. Use --resume or a different --run-id.")
    data = target / "photos"
    data.mkdir(parents=True)
    for index, photo in enumerate(photos):
        (data / f"{index:03d}.jpg").write_bytes(photo)
    (target / "config.json").write_text(json.dumps(config, indent=2))
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=131072,
    volumes={ROOT: volume},
    timeout=7200,
    max_containers=1,
)
def train_model(run_id: str, resume: bool = False):
    import subprocess

    validate_run_id(run_id)
    volume.reload()
    target = RUNS / run_id
    config = TrainConfig(**json.loads((target / "config.json").read_text()))
    output = target / "lora"
    if not (BASE / "READY").exists():
        raise ValueError("Download the base model first.")
    if (target / "COMPLETE").exists():
        raise ValueError("This run is complete. Start a new run to train again.")
    if resume and not list(output.glob("checkpoint-*")):
        raise ValueError("No checkpoint to resume. Use a new run ID.")
    command = config.command(str(BASE), str(target / "photos"), str(output), resume)
    (target / "command.json").write_text(json.dumps(command, indent=2))
    volume.commit()
    try:
        # Inherit stdout/stderr so Modal captures the real training logs and exit code.
        subprocess.run(command, check=True)
        if not (output / "pytorch_lora_weights.safetensors").is_file():
            raise RuntimeError("Trainer finished without producing LoRA weights.")
        (target / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat())
    finally:
        # Preserve checkpoints even when the trainer exits with an error.
        volume.commit()
    return run_id


@app.cls(
    image=image,
    gpu="A100-80GB",
    memory=65536,
    volumes={ROOT: volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=600,
)
class Benben:
    run_id: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from diffusers import FluxPipeline

        validate_run_id(self.run_id)
        volume.reload()
        target = RUNS / self.run_id
        if not (target / "COMPLETE").exists():
            raise ValueError(f"Training run {self.run_id} is not complete.")
        self.pipe = FluxPipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.load_lora_weights(target / "lora")

    @modal.method()
    def generate(
        self,
        prompt: str,
        seed: int = 117,
        steps: int = 28,
        guidance: float = 3.5,
        width: int = 1024,
        height: int = 1024,
    ):
        import io
        import torch

        if not prompt.strip() or len(prompt) > 2000:
            raise ValueError("Enter a prompt of 1–2000 characters.")
        if not 1 <= steps <= 50 or not 0 <= guidance <= 10:
            raise ValueError("Invalid steps or guidance.")
        if width not in (512, 768, 1024) or height not in (512, 768, 1024):
            raise ValueError("Dimensions must be 512, 768 or 1024.")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 4294967295.")
        picture = self.pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cuda").manual_seed(seed),
            width=width,
            height=height,
        ).images[0]
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        return buffer.getvalue()


@app.function(
    image=web_image, secrets=[web_secret], max_containers=1, min_containers=0, scaledown_window=60
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import io
    import os
    import gradio as gr
    from fastapi import FastAPI
    from PIL import Image

    def generate(run_id, scene, seed, steps):
        validate_run_id(run_id)
        prompt = f"{SUBJECT}, {scene.strip()}"
        result = Benben(run_id=run_id).generate.remote(prompt, int(seed), int(steps))
        return Image.open(io.BytesIO(result)), f"Seed: {int(seed)} · Prompt: {prompt}"

    with gr.Blocks(title="Benben's World", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Benben’s World\nA little Maltese. Endless adventures.")
        run_id = gr.Textbox(label="Training run", value="benben-v1")
        scene = gr.Textbox(
            label="Imagine Benben…", value="wearing a tiny astronaut suit on the moon"
        )
        gr.Examples(
            [
                "sitting in a field of wildflowers, warm afternoon photography",
                "as a watercolor portrait, soft pastel colors",
                "sailing a tiny boat on a peaceful lake",
            ],
            inputs=scene,
        )
        with gr.Row():
            seed = gr.Number(label="Seed", value=117, precision=0, minimum=0, maximum=2**32 - 1)
            steps = gr.Slider(1, 50, value=28, step=1, label="Detail steps")
        button = gr.Button("Imagine Benben", variant="primary")
        result = gr.Image(label="Benben", type="pil")
        details = gr.Textbox(label="Generation details")
        button.click(generate, [run_id, scene, seed, steps], [result, details], concurrency_limit=1)
        gr.Markdown("The first image may take a few minutes while the model wakes up.")
    ui.queue(max_size=10, default_concurrency_limit=1)
    return gr.mount_gradio_app(
        FastAPI(), ui, path="/", auth=(os.environ["USERNAME"], os.environ["PASSWORD"])
    )


@app.local_entrypoint()
def train(
    photos: str = "photos",
    run_id: str = "",
    steps: int = 500,
    rank: int = 16,
    learning_rate: float = 1e-4,
    resolution: int = 512,
    resume: bool = False,
):
    """Validate/upload photos, cache the model, then train one saved run."""
    if resume and not run_id:
        raise ValueError("--resume requires --run-id.")
    run_id = validate_run_id(run_id or datetime.now(timezone.utc).strftime("benben-%Y%m%d-%H%M%S"))
    config = TrainConfig(steps=steps, rank=rank, learning_rate=learning_rate, resolution=resolution)
    if not resume:
        pictures = prepare_photos(photos)  # Fail locally before downloading or renting a GPU.
        print(f"Uploading {len(pictures)} photos for {run_id}.")
        upload_dataset.remote(run_id, pictures, config.to_dict())
    download_model.remote()
    print(f"Training {run_id}. Resume uses the saved configuration.")
    train_model.remote(run_id, resume)
    print(f"Done. Generate: uv run modal run modal_app.py::generate --run-id {run_id}")


@app.local_entrypoint()
def generate(
    run_id: str,
    prompt: str = "wearing a tiny astronaut suit on the moon",
    seed: int = 117,
    output: str = "outputs/benben.png",
):
    validate_run_id(run_id)
    full_prompt = f"{SUBJECT}, {prompt}"
    result = Benben(run_id=run_id).generate.remote(full_prompt, seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(result)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {"run_id": run_id, "prompt": full_prompt, "seed": seed, "steps": 28, "guidance": 3.5},
            indent=2,
        )
    )
    print(destination.resolve())

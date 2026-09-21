"""Reference-conditioned alternative: preserve Benben while editing the scene."""

import io
import json
from pathlib import Path

import modal
from modal_app import image, volume, hf_secret
from benben import REFERENCE_PHOTOS, build_edit_prompt

app = modal.App("benben-kontext")
MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
KONTEXT = Path("/data/kontext")
model_image = image.add_local_python_source("modal_app")
REFERENCES = REFERENCE_PHOTOS


@app.function(image=model_image, secrets=[hf_secret], volumes={"/data": volume}, timeout=1200)
def download():
    from huggingface_hub import snapshot_download

    volume.reload()
    if (KONTEXT / "READY").exists():
        return
    snapshot_download(
        MODEL,
        local_dir=KONTEXT,
        allow_patterns=["model_index.json", "scheduler/*", "transformer/*"],
        ignore_patterns=["*.bin", "*.pt"],
    )
    (KONTEXT / "READY").write_text(MODEL)
    volume.commit()


@app.cls(
    image=model_image,
    gpu="A100-80GB",
    memory=65536,
    volumes={"/data": volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=600,
)
class Editor:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import (
            FluxPipeline,
            FluxKontextPipeline,
            FluxTransformer2DModel,
            FlowMatchEulerDiscreteScheduler,
        )

        volume.reload()
        if not (KONTEXT / "READY").exists():
            raise ValueError("Download Kontext first.")
        shared = FluxPipeline.from_pretrained(
            "/data/base", transformer=None, torch_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            KONTEXT / "transformer", torch_dtype=torch.bfloat16
        )
        self.pipe = FluxKontextPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(KONTEXT / "scheduler"),
            vae=shared.vae,
            text_encoder=shared.text_encoder,
            text_encoder_2=shared.text_encoder_2,
            tokenizer=shared.tokenizer,
            tokenizer_2=shared.tokenizer_2,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    @modal.method()
    def edit(self, instruction: str, reference: str = "portrait", seed: int = 117):
        import torch
        from PIL import Image

        if reference not in REFERENCES or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Invalid reference or seed.")
        photo = Image.open(Path("/data/runs/benben-v2/photos") / REFERENCES[reference]).convert(
            "RGB"
        )
        prompt = build_edit_prompt(instruction)
        result = self.pipe(
            image=photo,
            prompt=prompt,
            guidance_scale=2.5,
            num_inference_steps=28,
            width=1024,
            height=1024,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return buffer.getvalue(), {
            "prompt": prompt,
            "reference": reference,
            "seed": seed,
            "steps": 28,
            "guidance": 2.5,
            "model": MODEL,
            "width": result.width,
            "height": result.height,
        }


@app.function(image=model_image, volumes={"/data": volume}, timeout=60, max_containers=1)
def reference_image(reference: str = "portrait"):
    if reference not in REFERENCES:
        raise ValueError("Unknown reference photo.")
    volume.reload()
    return (Path("/data/runs/benben-v2/photos") / REFERENCES[reference]).read_bytes()


@app.local_entrypoint()
def test():
    download.remote()
    target = Path("outputs/kontext")
    target.mkdir(parents=True, exist_ok=True)
    for name, instruction, reference, seed in [
        (
            "garden",
            "Change the background to a lush sunlit garden with pink flowers. Place the dog on the grass.",
            "portrait",
            117,
        ),
        (
            "astronaut",
            "Put this dog in a small white astronaut suit on the surface of the moon. Leave his head uncovered, with the ponytail visible.",
            "standing",
            42,
        ),
        (
            "watercolor",
            "Transform this photo into a delicate watercolor illustration on white paper.",
            "smiling",
            7,
        ),
    ]:
        picture, metadata = Editor().edit.remote(instruction, reference, seed)
        (target / f"{name}.png").write_bytes(picture)
        (target / f"{name}.json").write_text(json.dumps(metadata, indent=2))
        print("Completed Kontext test:", name, flush=True)

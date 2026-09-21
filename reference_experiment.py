"""Small image-conditioned identity test using the existing base weights."""

from pathlib import Path
import json
import modal
from modal_app import image, volume

app = modal.App("benben-reference-experiment")


@app.function(
    image=image.add_local_python_source("modal_app"),
    volumes={"/data": volume},
    gpu="A100-80GB",
    memory=65536,
    timeout=600,
    max_containers=1,
)
def compare_reference():
    import io
    import torch
    from PIL import Image
    from diffusers import FluxImg2ImgPipeline

    volume.reload()
    pipe = FluxImg2ImgPipeline.from_pretrained("/data/base", torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # 14.png was the seventh image in the sorted v2 upload.
    source = Image.open("/data/runs/benben-v2/photos/006.jpg").convert("RGB")
    prompt = "A realistic photograph of a small white Maltese dog with a single curved ponytail tied high on the top of his head, long silky white ears, big round dark eyes and a compact black nose, sitting in a sunny garden with pink flowers. Natural photographic detail."
    results = []
    for name, strength, adapter in [
        ("base-055", 0.55, False),
        ("base-07", 0.7, False),
        ("v2-200-055", 0.55, True),
    ]:
        if adapter:
            pipe.load_lora_weights("/data/runs/benben-v2/lora/checkpoint-200")
        im = pipe(
            image=source,
            prompt=prompt,
            width=768,
            height=768,
            strength=strength,
            generator=torch.Generator("cuda").manual_seed(117),
            num_inference_steps=40,
            guidance_scale=3.5,
        ).images[0]
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        results.append(
            (
                name,
                buffer.getvalue(),
                {
                    "prompt": prompt,
                    "strength": strength,
                    "source": "benben-v2/photos/006.jpg",
                    "adapter": adapter,
                    "seed": 117,
                },
            )
        )
        print("Completed", name, flush=True)
    return results


@app.local_entrypoint()
def main():
    target = Path("outputs/reference-test")
    target.mkdir(parents=True, exist_ok=True)
    for name, picture, meta in compare_reference.remote():
        (target / f"{name}.png").write_bytes(picture)
        (target / f"{name}.json").write_text(json.dumps(meta, indent=2))

"""Bounded diagnostic comparisons; never changes the deployed model."""

import json
import re
from pathlib import Path

import modal
from modal_app import image, volume

app = modal.App("benben-experiments")
experiment_image = image.add_local_python_source("modal_app")


def validate_spec(variants: list[dict], experiment: str):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment):
        raise ValueError("Invalid experiment name.")
    if not 1 <= len(variants) <= 12:
        raise ValueError("Compare between 1 and 12 images per run.")
    names = set()
    for variant in variants:
        name = variant["name"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("Comparison names must be unique plain identifiers.")
        names.add(name)
        adapter = variant.get("adapter")
        if adapter is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}/lora(?:/checkpoint-[0-9]+)?", adapter
        ):
            raise ValueError("Adapter must be a run's lora directory or checkpoint.")
        if not isinstance(variant.get("prompt"), str) or not 1 <= len(variant["prompt"]) <= 2000:
            raise ValueError("Provide a nonempty prompt of at most 2000 characters.")
        if not 0 <= variant.get("seed", 117) <= 2**32 - 1:
            raise ValueError("Invalid seed.")
        if not 0 < variant.get("scale", 1.0) <= 2:
            raise ValueError("Comparison scale must be between 0 and 2.")


@app.function(
    image=experiment_image,
    volumes={"/data": volume},
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=900,
    max_containers=1,
)
def compare(variants: list[dict], experiment: str):
    import io
    import torch
    from diffusers import FluxPipeline
    from safetensors import safe_open

    validate_spec(variants, experiment)
    volume.reload()
    pipe = FluxPipeline.from_pretrained("/data/base", torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    loaded = None
    results = []
    for variant in variants:
        adapter = variant.get("adapter")
        stats = {}
        if adapter != loaded:
            if loaded is not None:
                pipe.unload_lora_weights()
            if adapter is not None:
                path = Path("/data/runs") / adapter / "pytorch_lora_weights.safetensors"
                with safe_open(path, framework="pt", device="cpu") as sf:
                    keys = list(sf.keys())
                    bkeys = [k for k in keys if "lora_B" in k]
                    stats = {
                        "tensor_count": len(keys),
                        "metadata": sf.metadata(),
                        "b_mean_abs": sum(
                            sf.get_tensor(k).float().abs().mean().item() for k in bkeys
                        )
                        / len(bkeys),
                    }
                pipe.load_lora_weights(path.parent, adapter_name="benben")
            loaded = adapter
        if adapter is not None:
            pipe.set_adapters("benben", adapter_weights=variant.get("scale", 1.0))
        picture = pipe(
            variant["prompt"],
            generator=torch.Generator("cuda").manual_seed(variant.get("seed", 117)),
            width=768,
            height=768,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        result = {"variant": variant, "stats": stats}
        output = Path("/data/evaluations") / experiment
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{variant['name']}.png").write_bytes(buffer.getvalue())
        (output / f"{variant['name']}.json").write_text(json.dumps(result, indent=2))
        results.append((buffer.getvalue(), result))
        print("Completed", variant["name"], flush=True)
    volume.commit()
    return results


@app.local_entrypoint()
def evaluate(spec: str, name: str):
    variants = json.loads(Path(spec).read_text())
    validate_spec(variants, name)
    results = compare.remote(variants, name)
    output = Path("outputs") / name
    output.mkdir(parents=True, exist_ok=True)
    for picture, metadata in results:
        key = metadata["variant"]["name"]
        (output / f"{key}.png").write_bytes(picture)
        (output / f"{key}.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(results)} comparisons to {output}")

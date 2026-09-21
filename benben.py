"""Pure local configuration, dataset preparation and trainer command construction."""

import hashlib
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.1-dev"
DIFFUSERS_COMMIT = "0f252be0ed42006c125ef4429156cb13ae6c1d60"  # v0.35.1
TRIGGER = "benbenmaltese"
SUBJECT = f"{TRIGGER} the Maltese dog"


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens.")
    return value


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 500
    rank: int = 16
    learning_rate: float = 1e-4
    resolution: int = 512
    seed: int = 117

    def __post_init__(self):
        if not 1 <= self.steps <= 5000:
            raise ValueError("Choose 1–5000 training steps.")
        if self.rank not in (4, 8, 16, 32, 64):
            raise ValueError("Rank must be 4, 8, 16, 32 or 64.")
        if self.resolution not in (512, 768, 1024):
            raise ValueError("Resolution must be 512, 768 or 1024.")
        if not 0 < self.learning_rate <= 1e-3:
            raise ValueError("Learning rate must be positive and at most 0.001.")

    def command(self, model: str, data: str, output: str, resume: bool = False):
        flags = {
            "pretrained_model_name_or_path": model,
            "instance_data_dir": data,
            "output_dir": output,
            "instance_prompt": f"a photo of {SUBJECT}",
            "resolution": self.resolution,
            "rank": self.rank,
            "max_train_steps": self.steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "checkpointing_steps": 100,
            "checkpoints_total_limit": 3,
            "mixed_precision": "bf16",
            "report_to": "tensorboard",
        }
        cmd = [
            "accelerate",
            "launch",
            "--num_processes=1",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "--dynamo_backend=no",
            "/opt/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py",
        ]
        cmd += [f"--{key}={value}" for key, value in flags.items()]
        cmd += ["--gradient_checkpointing"]
        if resume:
            cmd += ["--resume_from_checkpoint=latest"]
        return cmd

    def to_dict(self):
        return asdict(self)


def prepare_photos(folder: str) -> list[bytes]:
    """Validate locally; normalize orientation and strip metadata before upload."""
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"Photo folder does not exist: {root}")
    files = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    )
    if not 3 <= len(files) <= 40:
        raise ValueError(f"Found {len(files)} photos; provide 3–40 (10–20 recommended).")
    result, seen = [], set()
    for path in files:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if min(image.size) < 512:
                raise ValueError(f"{path.name}: both dimensions must be at least 512 pixels.")
            image.thumbnail((2048, 2048))
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            clean = Image.new("RGB", image.size)
            clean.paste(image)
            buffer = io.BytesIO()
            clean.save(buffer, format="JPEG", quality=95)
            result.append(buffer.getvalue())
    if len(result) < 3:
        raise ValueError("At least 3 distinct photos are needed after removing duplicates.")
    return result

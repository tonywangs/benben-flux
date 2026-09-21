"""Pure local configuration, dataset preparation and trainer command construction."""

import hashlib
import io
import json
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
    lora_alpha: int = 4  # Preserve the original v1 config when loading old runs.
    captioned: bool = False
    center_crop: bool = False
    checkpointing_steps: int = 100
    max_sequence_length: int = 512

    def __post_init__(self):
        if not 1 <= self.lora_alpha <= 128 or self.checkpointing_steps < 1:
            raise ValueError("Invalid LoRA alpha or checkpoint interval.")
        if not 77 <= self.max_sequence_length <= 512:
            raise ValueError("Sequence length must be between 77 and 512.")
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
            ("dataset_name" if self.captioned else "instance_data_dir"): data,
            "output_dir": output,
            "instance_prompt": f"a photo of {SUBJECT}",
            "resolution": self.resolution,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "max_sequence_length": self.max_sequence_length,
            "max_train_steps": self.steps,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "checkpointing_steps": self.checkpointing_steps,
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
        if self.captioned:
            cmd += ["--caption_column=text", "--image_column=image"]
        if self.center_crop:
            cmd += ["--center_crop"]
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


def prepare_captioned_photos(folder: str) -> tuple[list[bytes], list[str]]:
    """Validate caption-to-photo alignment before uploading a captioned dataset."""
    root = Path(folder).expanduser()
    rows = [
        json.loads(line)
        for line in (root / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    names = [r["file_name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate caption file names.")
    for row in rows:
        if Path(row["file_name"]).name != row["file_name"] or not row.get("text", "").strip():
            raise ValueError("Each caption needs a plain filename and nonempty text.")
    actual = sorted(
        p.name
        for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    )
    if set(actual) != set(names):
        raise ValueError("Captions must match every training photo exactly.")
    photos = prepare_photos(folder)
    if len(photos) != len(actual):
        raise ValueError("Remove duplicate photos and their captions before uploading.")
    captions = {r["file_name"]: r["text"] for r in rows}
    return photos, [captions[n] for n in actual]


REFERENCE_PHOTOS = {"portrait": "006.jpg", "standing": "001.jpg", "smiling": "004.jpg"}


def build_edit_prompt(instruction: str) -> str:
    if not instruction.strip() or len(instruction) > 1800:
        raise ValueError("Provide an edit instruction of 1–1800 characters.")
    return (
        instruction.strip() + " Preserve this exact dog’s identity, facial features, eye shape, "
        "black nose, long white ears and tied-up ponytail. Keep his ponytail clearly visible."
    )

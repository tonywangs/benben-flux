import io

import pytest
from PIL import Image

from benben import TrainConfig, prepare_photos, validate_run_id


def test_rank_and_steps_reach_trainer():
    command = TrainConfig(rank=32, steps=300).command("/base", "/photos", "/run", True)
    assert "--rank=32" in command
    assert "--max_train_steps=300" in command
    assert "--resume_from_checkpoint=latest" in command
    assert "--instance_prompt=a photo of benbenmaltese the Maltese dog" in command


@pytest.mark.parametrize("run_id", ["../base", "/base", "", "a/b", ".", "x" * 65])
def test_reject_unsafe_run_ids(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": 5001},
        {"rank": 0},
        {"learning_rate": float("nan")},
        {"resolution": 513},
    ],
)
def test_reject_invalid_training_settings(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_photos_are_rgb_oriented_and_metadata_free(tmp_path):
    for i, color in enumerate(["red", "green", "blue"]):
        img = Image.new("RGB", (600, 800), color)
        exif = Image.Exif()
        exif[274] = 6
        exif[270] = "private metadata"
        img.save(tmp_path / f"{i}.jpg", exif=exif)
    result = prepare_photos(str(tmp_path))
    assert len(result) == 3
    for payload in result:
        image = Image.open(io.BytesIO(payload))
        assert image.mode == "RGB"
        assert image.size == (800, 600)
        assert not image.getexif()


def test_duplicate_images_do_not_meet_minimum(tmp_path):
    for i in range(3):
        Image.new("RGB", (512, 512), "red").save(tmp_path / f"{i}.png")
    with pytest.raises(ValueError, match="distinct"):
        prepare_photos(str(tmp_path))


def test_small_image_rejected(tmp_path):
    for i in range(3):
        Image.new("RGB", (128, 512), "red").save(tmp_path / f"{i}.png")
    with pytest.raises(ValueError, match="512 pixels"):
        prepare_photos(str(tmp_path))


def test_missing_photos_fail_before_remote_work(tmp_path):
    with pytest.raises(ValueError, match="Found 0 photos"):
        prepare_photos(str(tmp_path))


def test_captioned_training_uses_dataset_and_explicit_alpha():
    cmd = TrainConfig(
        captioned=True,
        center_crop=True,
        lora_alpha=16,
        checkpointing_steps=200,
        max_sequence_length=256,
    ).command("/base", "/photos", "/run")
    assert "--dataset_name=/photos" in cmd
    assert not any(x.startswith("--instance_data_dir=") for x in cmd)
    assert "--caption_column=text" in cmd
    assert "--lora_alpha=16" in cmd
    assert "--center_crop" in cmd
    assert "--checkpointing_steps=200" in cmd


def test_caption_alignment_follows_uploaded_photo_order(tmp_path):
    import json
    from benben import prepare_captioned_photos

    rows = []
    for name, color in [("b.png", "blue"), ("a.png", "red"), ("c.png", "green")]:
        Image.new("RGB", (768, 768), color).save(tmp_path / name)
        rows.append({"file_name": name, "text": f"caption for {name}"})
    (tmp_path / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    photos, captions = prepare_captioned_photos(str(tmp_path))
    assert len(photos) == 3
    assert captions == ["caption for a.png", "caption for b.png", "caption for c.png"]
    rows.pop()
    (tmp_path / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="match every"):
        prepare_captioned_photos(str(tmp_path))


def test_reference_prompt_preserves_identity_and_rejects_empty_edits():
    from benben import build_edit_prompt

    prompt = build_edit_prompt("  Put him in a garden.  ")
    assert prompt.startswith("Put him in a garden.")
    assert "facial features" in prompt
    assert "pony" in prompt
    with pytest.raises(ValueError):
        build_edit_prompt("   ")
    with pytest.raises(ValueError):
        build_edit_prompt("x" * 1801)

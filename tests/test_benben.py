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

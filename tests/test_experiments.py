import pytest
from experiments import validate_spec


def test_evaluation_rejects_arbitrary_volume_paths():
    with pytest.raises(ValueError, match="Adapter"):
        validate_spec([{"name": "test", "prompt": "dog", "adapter": "../../base"}], "test")


def test_evaluation_rejects_duplicate_outputs_and_unbounded_batches():
    item = {"name": "same", "prompt": "dog", "adapter": None}
    with pytest.raises(ValueError, match="unique"):
        validate_spec([item, item], "test")
    with pytest.raises(ValueError, match="12"):
        validate_spec([item] * 13, "test")


def test_checked_in_evaluation_specs_are_valid():
    from pathlib import Path
    import json

    for path in Path("eval_specs").glob("*.json"):
        validate_spec(json.loads(path.read_text()), path.stem)

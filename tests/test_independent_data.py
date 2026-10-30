import json
from pathlib import Path

import pytest
from PIL import Image

from forgesight_ad.independent_data import DatasetManifestError, validate_manifest
from tools.evaluate_independent_dataset import aligned_binary_mask, summarize, wilson_interval


def _image(path: Path, size=(12, 9), color=(20, 30, 40)):
    Image.new("RGB", size, color).save(path)


def _manifest(root: Path, duplicate=False):
    images = root / "images"; masks = root / "masks"; images.mkdir(); masks.mkdir()
    rows = {}
    for index, split in enumerate(("fitting", "calibration", "development", "final_evaluation")):
        image = images / f"{split}.png"; _image(image, color=(20 + index, 30, 40))
        rows[split] = [{"image": str(image.relative_to(root)), "image_label": "normal"}]
    defect = images / "final_defect.png"; _image(defect, color=(90, 30, 40))
    mask = masks / "final_defect.png"; Image.new("L", (12, 9), 255).save(mask)
    rows["final_evaluation"].append({"image": str(defect.relative_to(root)), "mask": str(mask.relative_to(root)), "image_label": "defect", "defect_subtype": "scratch"})
    if duplicate:
        rows["calibration"][0]["image"] = rows["fitting"][0]["image"]
    payload = {"schema_version": "forgesight-independent-dataset-v1", "dataset_id": "fixture", "final_evaluation_immutable": True, "final_evaluation_sealed_at": "2026-10-31T00:00:00Z", "provenance": {"independent_from_historical_mvtec_tests": True}, "splits": rows}
    path = root / "manifest.json"; path.write_text(json.dumps(payload), encoding="utf-8"); return path


def test_independent_manifest_validates_hashes_masks_and_split_contract(tmp_path):
    report = validate_manifest(_manifest(tmp_path))
    assert report["dataset_id"] == "fixture"
    assert len(report["splits"]["final_evaluation"]) == 2
    assert report["splits"]["final_evaluation"][1]["mask_sha256"]


def test_independent_manifest_rejects_duplicate_content_across_splits(tmp_path):
    with pytest.raises(DatasetManifestError, match="Duplicate image content"):
        validate_manifest(_manifest(tmp_path, duplicate=True))


def test_independent_manifest_rejects_source_group_leakage(tmp_path):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["splits"]["fitting"][0]["source_group"] = "capture-1"
    payload["splits"]["calibration"][0]["source_group"] = "capture-1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="Source group crosses"):
        validate_manifest(path)


def test_final_evaluation_metrics_and_wilson_intervals_are_defined_from_frozen_predictions():
    rows = [
        {"image_label": 0, "score": 0.1, "prediction": "PASS", "mask": [[0, 0]], "predicted_mask": [[0, 0]], "anomaly_map": [[0.1, 0.2]], "total_call_ms": 1.0},
        {"image_label": 1, "score": 0.9, "prediction": "FAIL", "mask": [[1, 0]], "predicted_mask": [[1, 0]], "anomaly_map": [[0.9, 0.2]], "total_call_ms": 2.0},
    ]
    metrics = summarize(rows)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 0, "tp": 1}
    assert metrics["pixel_iou"] == 1.0
    assert wilson_interval(0, 10)[1] is not None


def test_non_square_ground_truth_mask_uses_detector_letterboxing_not_square_distortion():
    # VisA capsules are 1500x1000. A full foreground mask must retain the
    # 2:3 content area and have the same top/bottom padding as detector input.
    mask = Image.new("L", (1500, 1000), 255)
    aligned = aligned_binary_mask(mask)
    assert aligned.shape == (512, 512)
    assert aligned[0].sum() == 0
    assert aligned[-1].sum() == 0
    assert aligned[256].sum() == 512

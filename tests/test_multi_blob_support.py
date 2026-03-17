import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.nuscenes_metadata as nuscenes_metadata
from src.trainval_subset import build_trainval_camera_subset


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class MultiBlobSupportTests(unittest.TestCase):
    def test_build_trainval_camera_subset_unions_multiple_blob_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_dir = root / "metadata" / "v1.0-trainval"
            blob_root_06 = root / "part06"
            blob_root_10 = root / "part10"
            output_meta_dir = root / "derived" / "v1.0-trainval"
            image_csv_path = root / "csvdata" / "subset.csv"
            scene_token_path = root / "derived" / "scene_tokens.json"
            report_path = root / "derived" / "subset_report.json"

            (blob_root_06 / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
            (blob_root_10 / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
            (blob_root_06 / "samples" / "CAM_FRONT" / "frame1.jpg").write_bytes(b"frame1")
            (blob_root_10 / "samples" / "CAM_FRONT" / "frame2.jpg").write_bytes(b"frame2")

            write_json(
                meta_dir / "scene.json",
                [
                    {"token": "scene-1", "log_token": "log-1", "name": "scene-1", "description": "day scene"},
                    {"token": "scene-2", "log_token": "log-2", "name": "scene-2", "description": "rainy scene"},
                ],
            )
            write_json(
                meta_dir / "sample.json",
                [
                    {"token": "sample-1", "scene_token": "scene-1"},
                    {"token": "sample-2", "scene_token": "scene-2"},
                ],
            )
            write_json(
                meta_dir / "sample_data.json",
                [
                    {
                        "token": "sd-1",
                        "sample_token": "sample-1",
                        "filename": "samples/CAM_FRONT/frame1.jpg",
                        "is_key_frame": True,
                    },
                    {
                        "token": "sd-2",
                        "sample_token": "sample-2",
                        "filename": "samples/CAM_FRONT/frame2.jpg",
                        "is_key_frame": True,
                    },
                ],
            )
            write_json(meta_dir / "sample_annotation.json", [])
            write_json(meta_dir / "instance.json", [])
            write_json(meta_dir / "category.json", [])
            write_json(
                meta_dir / "log.json",
                [
                    {"token": "log-1", "location": "boston-seaport"},
                    {"token": "log-2", "location": "singapore-hollandvillage"},
                ],
            )

            report = build_trainval_camera_subset(
                dataset_roots=[blob_root_06, blob_root_10],
                source_meta_dir=meta_dir,
                output_meta_dir=output_meta_dir,
                image_csv_path=image_csv_path,
                scene_token_path=scene_token_path,
                report_path=report_path,
                image_id_start=2000,
            )

            self.assertEqual(report["scene_count"], 2)
            self.assertEqual(report["keyframe_count"], 2)
            self.assertEqual(sorted(report["dataset_roots"]), sorted([str(blob_root_06), str(blob_root_10)]))
            self.assertEqual(json.loads(scene_token_path.read_text(encoding="utf-8")), ["scene-1", "scene-2"])

            with image_csv_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows,
                [
                    {"id": "2000", "path": "samples/CAM_FRONT/frame1.jpg"},
                    {"id": "2001", "path": "samples/CAM_FRONT/frame2.jpg"},
                ],
            )

    def test_resolve_frame_path_checks_all_blob_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            blob_root_06 = root / "part06"
            blob_root_10 = root / "part10"
            target_path = blob_root_10 / "samples" / "CAM_FRONT" / "frame2.jpg"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(b"frame2")

            with patch.object(nuscenes_metadata, "NUSCENES_BLOB_ROOTS", [blob_root_06, blob_root_10]):
                resolved = nuscenes_metadata.resolve_frame_path(
                    "samples/CAM_FRONT/frame2.jpg",
                    metadata_index={"basename_to_sample_data": {}, "filename_to_sample_data": {}},
                )

            self.assertEqual(resolved, target_path)


if __name__ == "__main__":
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.trainval_subset import build_trainval_camera_subset, iter_json_array_records


def write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TrainvalSubsetTests(unittest.TestCase):
    def test_iter_json_array_records_streams_objects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "records.json"
            json_path.write_text('[{"a": 1}, {"b": 2}]', encoding="utf-8")

            records = list(iter_json_array_records(json_path, chunk_size=4))

        self.assertEqual(records, [{"a": 1}, {"b": 2}])

    def test_build_trainval_camera_subset_keeps_only_existing_camera_scene(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            source_meta_dir = dataset_root / "v1.0-trainval"
            output_meta_dir = root / "derived" / "v1.0-trainval"
            image_csv_path = root / "csvdata" / "subset.csv"
            scene_token_path = root / "derived" / "scene_tokens.json"
            report_path = root / "derived" / "subset_report.json"

            (dataset_root / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
            (dataset_root / "sweeps" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
            (dataset_root / "samples" / "CAM_FRONT" / "frame1.jpg").write_bytes(b"frame1")
            (dataset_root / "sweeps" / "CAM_FRONT" / "frame1_sweep.jpg").write_bytes(b"frame1s")

            write_json(
                source_meta_dir / "scene.json",
                [
                    {"token": "scene-1", "log_token": "log-1", "name": "scene-1", "description": "Night rain"},
                    {"token": "scene-2", "log_token": "log-2", "name": "scene-2", "description": "Sunny day"},
                ],
            )
            write_json(
                source_meta_dir / "sample.json",
                [
                    {"token": "sample-1", "scene_token": "scene-1"},
                    {"token": "sample-2", "scene_token": "scene-2"},
                ],
            )
            write_json(
                source_meta_dir / "sample_data.json",
                [
                    {
                        "token": "sd-1",
                        "sample_token": "sample-1",
                        "filename": "samples/CAM_FRONT/frame1.jpg",
                        "is_key_frame": True,
                    },
                    {
                        "token": "sd-2",
                        "sample_token": "sample-1",
                        "filename": "sweeps/CAM_FRONT/frame1_sweep.jpg",
                        "is_key_frame": False,
                    },
                    {
                        "token": "sd-3",
                        "sample_token": "sample-2",
                        "filename": "samples/CAM_FRONT/frame2.jpg",
                        "is_key_frame": True,
                    },
                ],
            )
            write_json(
                source_meta_dir / "sample_annotation.json",
                [
                    {"token": "ann-1", "sample_token": "sample-1", "instance_token": "instance-1"},
                    {"token": "ann-2", "sample_token": "sample-2", "instance_token": "instance-2"},
                ],
            )
            write_json(
                source_meta_dir / "instance.json",
                [
                    {"token": "instance-1", "category_token": "category-1"},
                    {"token": "instance-2", "category_token": "category-2"},
                ],
            )
            write_json(
                source_meta_dir / "category.json",
                [
                    {"token": "category-1", "name": "vehicle.car"},
                    {"token": "category-2", "name": "human.pedestrian.adult"},
                ],
            )
            write_json(
                source_meta_dir / "log.json",
                [
                    {"token": "log-1", "location": "boston-seaport"},
                    {"token": "log-2", "location": "singapore-hollandvillage"},
                ],
            )

            report = build_trainval_camera_subset(
                dataset_root=dataset_root,
                source_meta_dir=source_meta_dir,
                output_meta_dir=output_meta_dir,
                image_csv_path=image_csv_path,
                scene_token_path=scene_token_path,
                report_path=report_path,
                image_id_start=2000,
            )

            self.assertEqual(report["scene_count"], 1)
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["sample_data_count"], 2)
            self.assertEqual(report["keyframe_count"], 1)
            self.assertEqual(json.loads(scene_token_path.read_text(encoding="utf-8")), ["scene-1"])

            with image_csv_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"id": "2000", "path": "samples/CAM_FRONT/frame1.jpg"}])


if __name__ == "__main__":
    unittest.main()

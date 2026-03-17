from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from config import (
    IMAGE_CSV_PATH,
    IMAGE_ID_MIN,
    NUSCENES_BLOB_ROOTS,
    NUSCENES_META_DIR,
    NUSCENES_META_SOURCE_DIR,
    NUSCENES_SCENE_TOKEN_PATH,
    NUSCENES_SUBSET_REPORT_PATH,
)
from src.trainval_subset import build_trainval_camera_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the configured trainval camera-only metadata subset and image CSV.")
    parser.add_argument("--dataset-root", type=pathlib.Path, action="append", dest="dataset_roots")
    parser.add_argument("--source-meta-dir", type=pathlib.Path, default=NUSCENES_META_SOURCE_DIR)
    parser.add_argument("--output-meta-dir", type=pathlib.Path, default=NUSCENES_META_DIR)
    parser.add_argument("--image-csv", type=pathlib.Path, default=IMAGE_CSV_PATH)
    parser.add_argument("--scene-token-path", type=pathlib.Path, default=NUSCENES_SCENE_TOKEN_PATH)
    parser.add_argument("--report-path", type=pathlib.Path, default=NUSCENES_SUBSET_REPORT_PATH)
    parser.add_argument("--image-id-start", type=int, default=IMAGE_ID_MIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = args.dataset_roots or list(NUSCENES_BLOB_ROOTS)
    report = build_trainval_camera_subset(
        dataset_roots=dataset_roots,
        source_meta_dir=args.source_meta_dir,
        output_meta_dir=args.output_meta_dir,
        image_csv_path=args.image_csv,
        scene_token_path=args.scene_token_path,
        report_path=args.report_path,
        image_id_start=args.image_id_start,
    )
    print("Prepared configured trainval camera subset.")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import ENGCLIP_MODEL_DIR, IMAGE_CSV_PATH
from src.milvus_utils import REQUIRED_METADATA_FIELDS, get_or_create_collection, schema_needs_rebuild
from src.nuscenes_metadata import get_image_metadata, load_nuscenes_metadata, read_image_rows


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = 32
INSERT_FIELD_ORDER = ["id", "embedding", *REQUIRED_METADATA_FIELDS]


def batch_records(records, batch_size: int):
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_clip_model() -> tuple[CLIPProcessor, CLIPModel]:
    processor = CLIPProcessor.from_pretrained(str(ENGCLIP_MODEL_DIR))
    model = CLIPModel.from_pretrained(str(ENGCLIP_MODEL_DIR)).to(DEVICE).eval()
    return processor, model


def ensure_collection(drop_existing: bool):
    collection = get_or_create_collection(drop_existing=drop_existing)
    if schema_needs_rebuild(collection):
        raise RuntimeError("Milvus collection still uses the legacy schema. Re-run insert_image.py with --drop-existing once before reinserting data.")
    return collection


def iter_image_records(csv_path: Path, limit: int | None = None):
    metadata_index = load_nuscenes_metadata()
    emitted = 0
    skipped = 0
    for row_id, raw_path in read_image_rows(csv_path):
        metadata = get_image_metadata(raw_path, metadata_index)
        resolved_path = metadata.pop("resolved_path", "")
        if not resolved_path:
            skipped += 1
            continue

        yield {
            "id": int(row_id),
            "resolved_path": resolved_path,
            **metadata,
        }
        emitted += 1
        if limit is not None and emitted >= limit:
            break

    if skipped:
        print(f"Skipped {skipped} image rows because the frame file could not be resolved.")


def encode_image_batch(batch: list[dict], processor: CLIPProcessor, model: CLIPModel) -> np.ndarray:
    images = []
    for record in batch:
        with Image.open(record["resolved_path"]) as image:
            images.append(image.convert("RGB"))

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.detach().cpu().numpy().astype("float32")


def build_insert_payload(batch: list[dict], embeddings: np.ndarray) -> list[list]:
    payload = {
        "id": [record["id"] for record in batch],
        "embedding": embeddings.tolist(),
    }
    for field_name in REQUIRED_METADATA_FIELDS:
        payload[field_name] = [str(record.get(field_name, "") or "") for record in batch]
    return [payload[field_name] for field_name in INSERT_FIELD_ORDER]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insert nuScenes image embeddings and metadata into Milvus")
    parser.add_argument("--csv", type=Path, default=IMAGE_CSV_PATH, help="Path to the image CSV file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of images to encode per batch")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    parser.add_argument("--drop-existing", action="store_true", help="Drop and recreate the Milvus collection before inserting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Image CSV not found: {args.csv}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    collection = ensure_collection(args.drop_existing)
    processor, model = load_clip_model()

    total_inserted = 0
    for batch_index, batch in enumerate(batch_records(iter_image_records(args.csv, args.limit), args.batch_size), start=1):
        embeddings = encode_image_batch(batch, processor, model)
        collection.insert(build_insert_payload(batch, embeddings))
        total_inserted += len(batch)
        print(f"Inserted image batch {batch_index}: {len(batch)} rows (total={total_inserted}).")

    collection.flush()
    collection.load()
    print(f"Image insert complete on {DEVICE.type}. Collection now has {collection.num_entities} entities.")


if __name__ == "__main__":
    main()

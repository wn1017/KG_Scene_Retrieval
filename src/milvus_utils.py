from __future__ import annotations

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from config import (
    MILVUS_COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC_TYPE,
    MILVUS_NLIST,
    MILVUS_NPROBE,
    MILVUS_PORT,
    MILVUS_VECTOR_DIM,
)


SCHEMA_FIELD_DEFINITIONS = [
    FieldSchema(name="id", dtype=DataType.INT64, description="primary identifier", is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="normalized CLIP embedding", dim=MILVUS_VECTOR_DIM),
    FieldSchema(name="scene_token", dtype=DataType.VARCHAR, description="nuScenes scene token", max_length=64),
    FieldSchema(name="sample_token", dtype=DataType.VARCHAR, description="nuScenes sample token", max_length=64),
    FieldSchema(name="camera", dtype=DataType.VARCHAR, description="camera channel", max_length=32),
    FieldSchema(name="frame_path", dtype=DataType.VARCHAR, description="relative frame path", max_length=512),
    FieldSchema(name="weather", dtype=DataType.VARCHAR, description="weather label", max_length=32),
    FieldSchema(name="timeofday", dtype=DataType.VARCHAR, description="time of day label", max_length=32),
    FieldSchema(name="location", dtype=DataType.VARCHAR, description="location label", max_length=128),
    FieldSchema(name="obj_types", dtype=DataType.VARCHAR, description="comma separated object types", max_length=512),
]

REQUIRED_METADATA_FIELDS = [
    "scene_token",
    "sample_token",
    "camera",
    "frame_path",
    "weather",
    "timeofday",
    "location",
    "obj_types",
]

INDEX_PARAMS = {
    "metric_type": MILVUS_METRIC_TYPE,
    "index_type": MILVUS_INDEX_TYPE,
    "params": {"nlist": MILVUS_NLIST},
}

SEARCH_PARAMS = {
    "metric_type": MILVUS_METRIC_TYPE,
    "params": {"nprobe": MILVUS_NPROBE},
}


def connect_milvus():
    return connections.connect(alias="default", host=MILVUS_HOST, port=str(MILVUS_PORT))


def get_schema_field_names() -> list[str]:
    return [field.name for field in SCHEMA_FIELD_DEFINITIONS]


def get_collection(collection_name: str = MILVUS_COLLECTION_NAME) -> Collection | None:
    connect_milvus()
    if not utility.has_collection(collection_name):
        return None
    return Collection(collection_name)


def create_milvus_collection(
    collection_name: str = MILVUS_COLLECTION_NAME,
    dim: int = MILVUS_VECTOR_DIM,
    drop_existing: bool = False,
) -> Collection:
    connect_milvus()

    if utility.has_collection(collection_name):
        if drop_existing:
            utility.drop_collection(collection_name)
        else:
            return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, description="primary identifier", is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="normalized CLIP embedding", dim=dim),
        *SCHEMA_FIELD_DEFINITIONS[2:],
    ]
    schema = CollectionSchema(fields=fields, description="KG scene retrieval multimodal collection")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
    return collection


def get_or_create_collection(drop_existing: bool = False) -> Collection:
    collection = create_milvus_collection(drop_existing=drop_existing)
    try:
        collection.load()
    except Exception:
        pass
    return collection


def list_collection_fields(collection: Collection | None) -> list[str]:
    if collection is None:
        return []
    return [field.name for field in collection.schema.fields]


def has_field(collection: Collection | None, field_name: str) -> bool:
    return field_name in list_collection_fields(collection)


def get_search_output_fields(collection: Collection | None) -> list[str]:
    available_fields = set(list_collection_fields(collection))
    return [field for field in REQUIRED_METADATA_FIELDS if field in available_fields]


def schema_needs_rebuild(collection: Collection | None) -> bool:
    if collection is None:
        return False
    field_names = set(list_collection_fields(collection))
    return any(field not in field_names for field in REQUIRED_METADATA_FIELDS)


try:
    collection = get_or_create_collection(drop_existing=False)
except Exception:
    collection = None

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, NUSCENES_META_DIR


def read_json_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_object_name(category_name: str) -> str:
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("movable_object.trafficcone"):
        return "traffic_cone"
    if category_name.startswith("movable_object.barrier"):
        return "barrier"
    if category_name.startswith("animal"):
        return "animal"
    return category_name.replace(".", "_")


def infer_weather(description: str) -> str:
    lowered = description.lower()
    if any(token in lowered for token in ["rain", "wet"]):
        return "rainy"
    if "fog" in lowered:
        return "foggy"
    return "clear"


def infer_time_of_day(description: str) -> str:
    lowered = description.lower()
    if "night" in lowered or "lighting" in lowered:
        return "night"
    if "dusk" in lowered or "sunset" in lowered or "twilight" in lowered:
        return "dusk"
    return "day"


def infer_location_kind(description: str) -> str:
    lowered = description.lower()
    keyword_map = [
        ("intersection", ["intersection", "turn left", "cross intersection"]),
        ("crosswalk", ["crosswalk"]),
        ("parking_lot", ["parking lot"]),
        ("bus_stop", ["bus stop"]),
        ("sidewalk", ["sidewalk"]),
        ("highway", ["high speed", "highway", "freeway"]),
        ("street", ["street", "urban"]),
    ]
    for label, keywords in keyword_map:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "urban"


def load_metadata_tables(meta_dir: Path = NUSCENES_META_DIR) -> dict[str, list[dict]]:
    table_names = [
        "scene",
        "log",
        "sample",
        "sample_annotation",
        "instance",
        "category",
    ]
    return {name: read_json_records(meta_dir / f"{name}.json") for name in table_names}


def build_scene_records(meta_dir: Path = NUSCENES_META_DIR) -> list[dict]:
    tables = load_metadata_tables(meta_dir)

    scenes = tables["scene"]
    logs = {record["token"]: record for record in tables["log"]}
    sample_to_scene = {record["token"]: record["scene_token"] for record in tables["sample"]}
    instance_to_category = {record["token"]: record["category_token"] for record in tables["instance"]}
    category_names = {record["token"]: record["name"] for record in tables["category"]}

    scene_to_instances: dict[str, dict[str, str]] = defaultdict(dict)
    for annotation in tables["sample_annotation"]:
        scene_token = sample_to_scene.get(annotation["sample_token"])
        if not scene_token:
            continue

        category_token = instance_to_category.get(annotation["instance_token"])
        if not category_token:
            continue

        object_name = normalize_object_name(category_names[category_token])
        scene_to_instances[scene_token][annotation["instance_token"]] = object_name

    records: list[dict] = []
    for scene in scenes:
        description = scene.get("description", "")
        log_record = logs.get(scene.get("log_token", ""), {})
        location_area = log_record.get("location", "unknown")
        location_kind = infer_location_kind(description)
        location_key = f"{location_area}|{location_kind}"

        object_counter = Counter(scene_to_instances.get(scene["token"], {}).values())
        object_counts = dict(sorted(object_counter.items(), key=lambda item: (-item[1], item[0])))

        records.append(
            {
                "scene_token": scene["token"],
                "scene_name": scene.get("name", ""),
                "description": description,
                "num_samples": int(scene.get("nbr_samples", 0) or 0),
                "weather": infer_weather(description),
                "timeofday": infer_time_of_day(description),
                "location_area": location_area,
                "location_kind": location_kind,
                "location_key": location_key,
                "objects": object_counts,
            }
        )

    return records


def filter_scene_records(
    records: Iterable[dict],
    weather: str | None = None,
    timeofday: str | None = None,
    object_types: list[str] | None = None,
    location_kind: str | None = None,
) -> list[str]:
    object_types = object_types or []
    matched_tokens: list[str] = []

    for record in records:
        if weather and record["weather"] != weather:
            continue
        if timeofday and record["timeofday"] != timeofday:
            continue
        if location_kind and record["location_kind"] != location_kind:
            continue
        if object_types and not all(obj in record["objects"] for obj in object_types):
            continue
        matched_tokens.append(record["scene_token"])

    return matched_tokens


def get_neo4j_driver():
    auth = (NEO4J_USER, NEO4J_PASSWORD) if NEO4J_USER or NEO4J_PASSWORD else None
    return GraphDatabase.driver(NEO4J_URI, auth=auth)


def create_constraints(session) -> None:
    session.run("CREATE CONSTRAINT scene_token_if_not_exists IF NOT EXISTS FOR (s:Scene) REQUIRE s.scene_token IS UNIQUE")
    session.run("CREATE CONSTRAINT weather_name_if_not_exists IF NOT EXISTS FOR (w:Weather) REQUIRE w.name IS UNIQUE")
    session.run("CREATE CONSTRAINT time_name_if_not_exists IF NOT EXISTS FOR (t:TimeOfDay) REQUIRE t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT object_name_if_not_exists IF NOT EXISTS FOR (o:Object) REQUIRE o.name IS UNIQUE")
    session.run("CREATE CONSTRAINT location_key_if_not_exists IF NOT EXISTS FOR (l:Location) REQUIRE l.key IS UNIQUE")


def _raise_neo4j_unavailable(exc: Exception) -> None:
    raise RuntimeError(f"Neo4j unavailable: {exc}") from exc


def write_scene_graph(records: list[dict], rebuild: bool = False) -> None:
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            create_constraints(session)
            if rebuild:
                session.run("MATCH ()-[r]->() DELETE r")
                session.run("MATCH (n) DELETE n")
                create_constraints(session)

            for record in records:
                session.run(
                    """
                    MERGE (s:Scene {scene_token: $scene_token})
                    SET s.name = $scene_name,
                        s.description = $description,
                        s.num_samples = $num_samples,
                        s.weather = $weather,
                        s.timeofday = $timeofday,
                        s.location_area = $location_area,
                        s.location_kind = $location_kind
                    MERGE (w:Weather {name: $weather})
                    MERGE (t:TimeOfDay {name: $timeofday})
                    MERGE (l:Location {key: $location_key})
                    SET l.name = $location_area,
                        l.kind = $location_kind,
                        l.area = $location_area
                    MERGE (s)-[:WEATHER]->(w)
                    MERGE (s)-[:TIMEOFDAY]->(t)
                    MERGE (s)-[:LOCTYPE]->(l)
                    """,
                    record,
                )

                for object_name, count in record["objects"].items():
                    session.run(
                        """
                        MATCH (s:Scene {scene_token: $scene_token})
                        MERGE (o:Object {name: $object_name})
                        MERGE (s)-[r:CONTAINS]->(o)
                        SET r.count = $count
                        """,
                        {
                            "scene_token": record["scene_token"],
                            "object_name": object_name,
                            "count": int(count),
                        },
                    )

            area_to_tokens: dict[str, list[str]] = defaultdict(list)
            for record in records:
                area_to_tokens[record["location_area"]].append(record["scene_token"])

            for area, tokens in area_to_tokens.items():
                for left_token, right_token in combinations(sorted(tokens), 2):
                    session.run(
                        """
                        MATCH (a:Scene {scene_token: $left_token})
                        MATCH (b:Scene {scene_token: $right_token})
                        MERGE (a)-[:NEARBY {reason: $reason}]->(b)
                        MERGE (b)-[:NEARBY {reason: $reason}]->(a)
                        """,
                        {"left_token": left_token, "right_token": right_token, "reason": f"same_area:{area}"},
                    )
    except (ServiceUnavailable, Neo4jError) as exc:
        _raise_neo4j_unavailable(exc)
    finally:
        driver.close()


def query_scene_tokens(
    weather: str | None = None,
    timeofday: str | None = None,
    object_types: list[str] | None = None,
    location_kind: str | None = None,
) -> list[str]:
    object_types = object_types or []
    driver = get_neo4j_driver()
    cypher = """
    MATCH (s:Scene)
    WHERE ($weather IS NULL OR EXISTS { MATCH (s)-[:WEATHER]->(:Weather {name: $weather}) })
      AND ($timeofday IS NULL OR EXISTS { MATCH (s)-[:TIMEOFDAY]->(:TimeOfDay {name: $timeofday}) })
      AND ($location_kind IS NULL OR EXISTS { MATCH (s)-[:LOCTYPE]->(:Location {kind: $location_kind}) })
      AND (size($object_types) = 0 OR ALL(object_name IN $object_types WHERE EXISTS {
            MATCH (s)-[:CONTAINS]->(:Object {name: object_name})
      }))
    RETURN s.scene_token AS scene_token
    ORDER BY s.scene_token
    """
    try:
        with driver.session() as session:
            result = session.run(
                cypher,
                {
                    "weather": weather,
                    "timeofday": timeofday,
                    "location_kind": location_kind,
                    "object_types": object_types,
                },
            )
            tokens = [record["scene_token"] for record in result]
    except (ServiceUnavailable, Neo4jError) as exc:
        _raise_neo4j_unavailable(exc)
    finally:
        driver.close()
    return tokens


def print_summary(records: list[dict]) -> None:
    print(f"Built {len(records)} scene records from {NUSCENES_META_DIR}")
    rainy_night_tokens = filter_scene_records(records, weather="rainy", timeofday="night")
    print("Rainy AND night scene tokens:", rainy_night_tokens)
    for record in records:
        top_objects = list(record["objects"].items())[:5]
        print(
            f"{record['scene_name']} | weather={record['weather']} | time={record['timeofday']} | "
            f"location={record['location_area']}:{record['location_kind']} | top_objects={top_objects}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a nuScenes scene graph for Neo4j")
    parser.add_argument("--write", action="store_true", help="Write graph records into Neo4j")
    parser.add_argument("--rebuild", action="store_true", help="Clear the existing graph before writing")
    parser.add_argument("--query-weather", default=None)
    parser.add_argument("--query-time", default=None)
    parser.add_argument("--query-object", action="append", default=[])
    parser.add_argument("--query-location", default=None)
    args = parser.parse_args()

    records = build_scene_records()
    print_summary(records)

    if args.write:
        write_scene_graph(records, rebuild=args.rebuild)
        print("Graph data written to Neo4j.")

    if any([args.query_weather, args.query_time, args.query_object, args.query_location]):
        if args.write:
            tokens = query_scene_tokens(
                weather=args.query_weather,
                timeofday=args.query_time,
                object_types=args.query_object,
                location_kind=args.query_location,
            )
            print("Neo4j query scene tokens:", tokens)
        else:
            tokens = filter_scene_records(
                records,
                weather=args.query_weather,
                timeofday=args.query_time,
                object_types=args.query_object,
                location_kind=args.query_location,
            )
            print("Local filter scene tokens:", tokens)


if __name__ == "__main__":
    main()

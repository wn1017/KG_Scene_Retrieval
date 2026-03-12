import unittest

from neo4j.exceptions import ServiceUnavailable

import src.kg_builder as kg_builder


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *args, **kwargs):
        raise ServiceUnavailable("neo4j down")


class _FakeDriver:
    def session(self):
        return _FakeSession()

    def close(self):
        return None


class KgBuilderTests(unittest.TestCase):
    def test_filter_scene_records_finds_rainy_night_scene(self):
        records = kg_builder.build_scene_records()
        tokens = kg_builder.filter_scene_records(records, weather="rainy", timeofday="night")
        self.assertEqual(tokens, ["de7d80a1f5fb4c3e82ce8a4f213b450a"])

    def test_write_scene_graph_wraps_neo4j_connection_errors(self):
        original_get_driver = kg_builder.get_neo4j_driver
        kg_builder.get_neo4j_driver = lambda: _FakeDriver()
        try:
            with self.assertRaisesRegex(RuntimeError, "Neo4j"):
                kg_builder.write_scene_graph([{ 
                    "scene_token": "scene-token",
                    "scene_name": "scene-name",
                    "description": "Night and rain at intersection",
                    "num_samples": 1,
                    "weather": "rainy",
                    "timeofday": "night",
                    "location_area": "boston-seaport",
                    "location_kind": "intersection",
                    "location_key": "boston-seaport|intersection",
                    "objects": {"pedestrian": 2},
                }])
        finally:
            kg_builder.get_neo4j_driver = original_get_driver


if __name__ == "__main__":
    unittest.main()

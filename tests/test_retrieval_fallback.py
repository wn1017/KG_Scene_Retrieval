import unittest
from unittest.mock import patch

import numpy as np

import app


class RetrievalFallbackTests(unittest.TestCase):
    def setUp(self):
        app.clear_runtime_caches()

    def test_dynamic_retrieve_surfaces_fallback_status_in_visible_detail_panel(self):
        fake_result = (
            [],
            "English-CLIP",
            {"weather": "foggy", "time": "night", "objects": []},
            "Neo4j 未找到候选场景；当前 170-scene 子集没有满足 天气=foggy; 时段=night 的场景；已跳过知识图谱过滤，继续展示相似度检索结果。",
        )

        with patch.object(app, "retrieve_images", return_value=fake_result):
            output = app.dynamic_retrieve("foggy night", "text2image")

        self.assertIn("Neo4j 未找到候选场景", output[-1])
        self.assertIn("已跳过知识图谱过滤", output[-1])
        self.assertIn("继续展示相似度检索结果", output[-1])

    def test_get_candidate_scene_tokens_keeps_similarity_fallback_when_subset_has_no_match(self):
        parsed_query = app.parse_query("night")
        subset_records = [
            {
                "scene_token": f"scene-{index}",
                "scene_name": f"scene-{index}",
                "description": "day scene",
                "num_samples": 1,
                "weather": "clear",
                "timeofday": "day",
                "location_area": "boston-seaport",
                "location_kind": "urban",
                "location_key": f"boston-seaport|urban-{index}",
                "objects": {},
            }
            for index in range(85)
        ]

        with patch.object(app, "KG_SCENE_RECORDS", subset_records), patch.object(
            app, "query_scene_tokens", side_effect=RuntimeError("Neo4j unavailable")
        ):
            tokens, status, should_stop = app.get_candidate_scene_tokens(parsed_query)

        self.assertEqual(tokens, [])
        self.assertFalse(should_stop)
        self.assertIn("85-scene", status)
        self.assertIn("night", status)
        self.assertIn("相似度检索", status)

    def test_retrieve_images_falls_back_to_similarity_search_when_kg_has_no_candidates(self):
        subset_records = [
            {
                "scene_token": f"scene-{index}",
                "scene_name": f"scene-{index}",
                "description": "day scene",
                "num_samples": 1,
                "weather": "clear",
                "timeofday": "day",
                "location_area": "boston-seaport",
                "location_kind": "urban",
                "location_key": f"boston-seaport|urban-{index}",
                "objects": {},
            }
            for index in range(85)
        ]
        query_vector = np.zeros(4, dtype=np.float32)

        with patch.object(app, "KG_SCENE_RECORDS", subset_records), patch.object(
            app, "query_scene_tokens", side_effect=RuntimeError("Neo4j unavailable")
        ), patch.object(app, "encode_text_query", return_value=(query_vector, "English-CLIP")) as encode_mock, patch.object(
            app, "search_frame_hits", return_value=[]
        ) as search_mock:
            image_results, model_name, parsed_query, kg_status = app.retrieve_images("night")

        self.assertEqual(parsed_query["time"], "night")
        self.assertEqual(image_results, [])
        self.assertEqual(model_name, "English-CLIP")
        self.assertIn("相似度检索", kg_status)
        encode_mock.assert_called_once()
        search_mock.assert_called_once_with(query_vector, app.IMAGE_RESULT_COUNT, [])

    def test_retrieve_videos_falls_back_to_similarity_search_when_kg_has_no_candidates(self):
        subset_records = [
            {
                "scene_token": f"scene-{index}",
                "scene_name": f"scene-{index}",
                "description": "day scene",
                "num_samples": 1,
                "weather": "clear",
                "timeofday": "day",
                "location_area": "boston-seaport",
                "location_kind": "urban",
                "location_key": f"boston-seaport|urban-{index}",
                "objects": {},
            }
            for index in range(85)
        ]
        query_vector = np.zeros(4, dtype=np.float32)

        with patch.object(app, "KG_SCENE_RECORDS", subset_records), patch.object(
            app, "query_scene_tokens", side_effect=RuntimeError("Neo4j unavailable")
        ), patch.object(app, "encode_text_query", return_value=(query_vector, "English-CLIP")) as encode_mock, patch.object(
            app, "search_frame_hits", return_value=[]
        ) as search_mock:
            video_results, model_name, parsed_query, kg_status = app.retrieve_videos("night")

        self.assertEqual(parsed_query["time"], "night")
        self.assertEqual(video_results, [])
        self.assertEqual(model_name, "English-CLIP")
        self.assertIn("相似度检索", kg_status)
        encode_mock.assert_called_once()
        search_mock.assert_called_once_with(query_vector, app.VIDEO_SEARCH_LIMIT, [])


if __name__ == "__main__":
    unittest.main()

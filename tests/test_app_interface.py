import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app


class _FakeHit:
    def __init__(self, hit_id: int, score: float, entity: dict):
        self.id = hit_id
        self.score = score
        self.entity = entity


class _FakeCollection:
    def __init__(self):
        self.num_entities = 10
        self.search_calls: list[dict] = []
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return None

    def search(self, data, anns_field, param, limit, expr, output_fields):
        self.search_calls.append(
            {
                "data": data,
                "anns_field": anns_field,
                "param": param,
                "limit": limit,
                "expr": expr,
                "output_fields": output_fields,
            }
        )
        if "scene_token in" in expr:
            return [[]]
        return [[
            _FakeHit(101, 0.95, {"scene_token": "other-scene", "frame_path": "samples/CAM_FRONT/other.jpg"}),
            _FakeHit(202, 0.91, {"scene_token": "scene-a", "frame_path": "samples/CAM_FRONT/match.jpg"}),
        ]]


class AppInterfaceTests(unittest.TestCase):
    def setUp(self):
        app.clear_runtime_caches()

    def test_parse_query_collects_highlight_matches_for_original_terms(self):
        parsed = app.parse_query("夜间路口有汽车")
        match_pairs = {(item["text"], item["category"]) for item in parsed["highlight_matches"]}

        self.assertEqual(parsed["time"], "night")
        self.assertEqual(parsed["location"], "intersection")
        self.assertIn("car", parsed["objects"])
        self.assertIn(("夜间", "time"), match_pairs)
        self.assertIn(("路口", "location"), match_pairs)
        self.assertIn(("汽车", "object"), match_pairs)

    def test_build_query_highlight_html_renders_marker_markup(self):
        parsed = app.parse_query("rainy intersection with cars at night")
        html = app.build_query_highlight_html(parsed)

        self.assertIn("query-highlight-block", html)
        self.assertIn("总耗时 0.10s", html)
        self.assertIn("query-marker marker-weather", html)
        self.assertIn("query-marker marker-time", html)
        self.assertIn("query-marker marker-location", html)
        self.assertIn("query-marker marker-object", html)
        self.assertIn("query-highlight-chip", html)
        self.assertIn("rainy", html)
        self.assertIn("cars", html)

    def test_parse_query_matches_extended_object_aliases_without_generic_overlap(self):
        parsed = app.parse_query("police car beside a trailer and bicycle rack")

        self.assertIn("vehicle_emergency_police", parsed["objects"])
        self.assertIn("trailer", parsed["objects"])
        self.assertIn("static_object_bicycle_rack", parsed["objects"])
        self.assertNotIn("car", parsed["objects"])
        self.assertNotIn("bicycle", parsed["objects"])

    def test_parse_query_matches_extended_chinese_object_aliases(self):
        parsed = app.parse_query(
            "\u6551\u62a4\u8f66\u65c1\u8fb9\u6709\u8b66\u8f66\u3001\u62d6\u8f66\u3001\u81ea\u884c\u8f66\u67b6\u548c\u52a8\u7269"
        )

        self.assertIn("vehicle_emergency_ambulance", parsed["objects"])
        self.assertIn("vehicle_emergency_police", parsed["objects"])
        self.assertIn("trailer", parsed["objects"])
        self.assertIn("static_object_bicycle_rack", parsed["objects"])
        self.assertIn("animal", parsed["objects"])
        self.assertNotIn("car", parsed["objects"])
        self.assertNotIn("bicycle", parsed["objects"])

    def test_build_explanation_html_renders_top_runtime_bar_and_two_focus_cards(self):
        parsed_query = {
            "weather": "rainy",
            "time": "night",
            "objects": ["pedestrian"],
            "location": "intersection",
            "raw_text": "rainy night intersection with pedestrians",
            "highlight_matches": [
                {"start": 0, "end": 5, "text": "rainy", "category": "weather", "label": "rainy"},
                {"start": 6, "end": 11, "text": "night", "category": "time", "label": "night"},
                {"start": 12, "end": 24, "text": "intersection", "category": "location", "label": "intersection"},
                {"start": 30, "end": 41, "text": "pedestrians", "category": "object", "label": "pedestrian"},
            ],
        }
        html = app.build_explanation_html(
            parsed_query,
            "Neo4j filtered 1 scene | 耗时：NLP=12ms, 知识图谱=34ms, Milvus=56ms",
            "Chinese-CLIP",
            "text2image",
            3,
        )

        self.assertIn("query-highlight-block", html)
        self.assertIn("detail-panel", html)
        self.assertIn("runtime-strip", html)
        self.assertIn("runtime-total", html)
        self.assertIn("runtime-breakdown", html)
        self.assertIn("runtime-chip", html)
        self.assertIn("focus-card-grid", html)
        self.assertIn("focus-card nlp-focus-card", html)
        self.assertIn("focus-card kg-focus-card", html)
        self.assertIn("focus-card-body", html)
        self.assertIn("module-inline-details", html)
        self.assertIn("kg-path-block", html)
        self.assertIn("path-segment path-segment-relation", html)
        self.assertIn("Scene.scene_token", html)
        self.assertIn("CONTAINS {count}", html)
        self.assertIn("天气 / weather", html)
        self.assertIn("vehicle_emergency_ambulance", html)
        self.assertIn("night", html)
        self.assertIn("pedestrian", html)
        self.assertIn("intersection", html)
        self.assertIn("检索时间", html)
        self.assertIn("NLP=12ms", html)
        self.assertIn("query-marker marker-weather", html)
        self.assertNotIn("当前检索解析", html)
        self.assertNotIn("自然语言解析模块", html)
        self.assertNotIn("知识图谱过滤模块", html)
        self.assertNotIn("kg-runtime-meta", html)
        self.assertNotIn("path-summary-chip", html)
        self.assertNotIn("Chinese-CLIP", html)

    def test_build_explanation_html_places_query_highlight_inside_nlp_focus_card(self):
        parsed_query = {
            "time": "night",
            "location": "intersection",
            "objects": ["car"],
            "raw_text": "night intersection with cars",
            "highlight_matches": [
                {"start": 0, "end": 5, "text": "night", "category": "time", "label": "night"},
                {"start": 6, "end": 18, "text": "intersection", "category": "location", "label": "intersection"},
                {"start": 24, "end": 28, "text": "cars", "category": "object", "label": "car"},
            ],
        }

        html = app.build_explanation_html(
            parsed_query,
            "Neo4j filtered 0 scenes",
            "English-CLIP",
            "text2image",
            0,
        )

        highlight_index = html.index("query-highlight-block")
        nlp_card_index = html.index("focus-card nlp-focus-card")

        self.assertGreater(highlight_index, nlp_card_index)

    def test_build_explanation_html_keeps_empty_nlp_and_kg_cards_balanced(self):
        html = app.build_explanation_html({}, "尚未执行知识图谱过滤。", "待命", "text2image", 0)

        self.assertIn("focus-card-body", html)
        self.assertIn("query-highlight-empty", html)
        self.assertIn("kg-path-empty", html)
        self.assertIn("runtime-chip-muted", html)

    def test_build_image_caption_wraps_metadata_in_details(self):
        caption = app.build_image_caption(
            {
                "score": 0.9321,
                "camera": "CAM_FRONT",
                "scene_name": "scene-0061",
                "weather": "rainy",
                "timeofday": "night",
                "obj_types": "pedestrian,car",
                "location": "boston-seaport:intersection",
                "resolved_frame_path": str(app.NUSCENES_ROOT / "samples" / "CAM_FRONT" / "sample.jpg"),
            },
            {"weather": "rainy", "time": "night", "objects": ["pedestrian"], "location": "intersection"},
            "Neo4j filtered 1 scene",
            "Chinese-CLIP",
        )

        self.assertIn("<details", caption)
        self.assertIn("scene-0061", caption)
        self.assertIn("0.9321", caption)

    def test_dynamic_retrieve_accepts_five_image_slots_and_preview_reset_payload(self):
        fake_image = Image.new("RGB", (8, 8), color="white")
        fake_result = (
            [(fake_image, "caption")],
            "English-CLIP",
            {"weather": "rainy", "objects": ["pedestrian"]},
            "Neo4j filtered 1 scene",
        )

        with patch.object(app, "retrieve_images", return_value=fake_result):
            output = app.dynamic_retrieve("rainy night with pedestrians", "text2image")

        expected_output_count = app.IMAGE_RESULT_COUNT * 2 + app.VIDEO_RESULT_COUNT * 2 + 4
        self.assertEqual(len(output), expected_output_count)
        self.assertIsNotNone(output[0])
        self.assertEqual(output[app.IMAGE_RESULT_COUNT], "caption")
        self.assertIsNone(output[-4])
        self.assertEqual(output[-3], "")
        self.assertIsInstance(output[-2], dict)
        self.assertFalse(output[-2]["visible"])
        self.assertIn("detail-panel", output[-1])
        self.assertIn("focus-card", output[-1])
        self.assertNotIn("status-card", output[-1])
        self.assertNotIn("Chinese-CLIP", output[-1])

    def test_custom_css_keeps_five_column_images_with_custom_preview_button(self):
        self.assertNotIn("button:hover, .gr-button:hover", app.custom_css)
        self.assertIn("#search-btn:hover, #clear-btn:hover", app.custom_css)
        self.assertIn("grid-template-columns: repeat(5, minmax(0, 1fr));", app.custom_css)
        self.assertIn("body.lightbox-open", app.custom_css)
        self.assertIn("background: rgba(248, 250, 252", app.custom_css)
        self.assertIn("100dvh", app.custom_css)
        self.assertIn(".query-marker", app.custom_css)
        self.assertIn(".marker-time", app.custom_css)
        self.assertIn(".query-highlight-chip", app.custom_css)
        self.assertIn(".module-inline-details", app.custom_css)
        self.assertIn(".kg-path-block", app.custom_css)
        self.assertIn(".path-segment-relation", app.custom_css)
        self.assertIn(".focus-card-grid", app.custom_css)
        self.assertIn(".focus-card-body", app.custom_css)
        self.assertIn(".runtime-strip", app.custom_css)
        self.assertIn(".runtime-total", app.custom_css)
        self.assertIn(".runtime-breakdown", app.custom_css)
        self.assertNotIn("max-height: 132px;", app.custom_css)
        self.assertNotIn("overflow: auto;", app.custom_css)
        source = Path(app.__file__).read_text(encoding="utf-8")
        self.assertIn("gr.Image(", source)
        self.assertIn('gr.Button("放大查看"', source)
        self.assertIn("def open_image_preview", source)
        self.assertIn("document.body.classList.add('lightbox-open')", source)
        self.assertIn("document.body.classList.remove('lightbox-open')", source)
        self.assertNotIn("gr.Gallery(", source)
        self.assertEqual(app.MODE_LABELS["text2image"], "搜索图片")
        self.assertEqual(app.MODE_LABELS["text2video"], "搜索视频片段")
        self.assertIn("show_fullscreen_button=False", source)

    def test_prepare_video_render_frames_deduplicates_without_blended_transition_frame(self):
        temp_dir = Path(app.__file__).resolve().parent / ".tmp_test_video_frames"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        red_path = temp_dir / "red.png"
        blue_path = temp_dir / "blue.png"
        Image.new("RGB", (48, 32), color=(255, 0, 0)).save(red_path)
        Image.new("RGB", (48, 32), color=(0, 0, 255)).save(blue_path)

        try:
            frames = app.prepare_video_render_frames([red_path, red_path, blue_path])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(len(frames), 2)
        self.assertTrue(np.array_equal(frames[0][0, 0], np.array([0, 0, 255])))
        self.assertTrue(np.array_equal(frames[1][0, 0], np.array([255, 0, 0])))

    def test_write_video_clip_generates_browser_playable_media(self):
        self.assertTrue(app.CAMERA_SEQUENCES)
        sequence_key, sequence = next(iter(app.CAMERA_SEQUENCES.items()))
        scene_token, camera = sequence_key
        frame_paths = [
            app.NUSCENES_ROOT / item["filename"]
            for item in sequence
            if (app.NUSCENES_ROOT / item["filename"]).exists()
        ][: min(8, len(sequence))]

        self.assertGreaterEqual(len(frame_paths), 2)

        anchor_record = {
            "scene_token": scene_token,
            "scene_name": app.SCENE_BY_TOKEN.get(scene_token, {}).get("name", scene_token),
            "camera": camera,
        }

        clip_path = app.write_video_clip(anchor_record, frame_paths)

        self.assertIsNotNone(clip_path)
        clip_path = Path(clip_path)
        self.assertTrue(clip_path.exists())
        self.assertTrue(app.is_browser_playable_clip(clip_path))
        clip_path.unlink(missing_ok=True)

    def test_get_candidate_scene_tokens_falls_back_to_local_kg_when_neo4j_returns_unknown_tokens(self):
        parsed_query = {
            "weather": "rainy",
            "time": "night",
            "objects": ["pedestrian"],
            "location": "intersection",
        }

        with patch.object(app, "query_scene_tokens", return_value=["scene-token"]), patch.object(
            app, "get_local_candidate_scene_tokens", return_value=["scene-a"]
        ):
            tokens, status, should_stop = app.get_candidate_scene_tokens(parsed_query)

        self.assertEqual(tokens, ["scene-a"])
        self.assertIn("Neo4j", status)
        self.assertTrue(status)
        self.assertFalse(should_stop)

    def test_get_candidate_scene_tokens_reports_zero_candidates_when_no_valid_tokens_exist(self):
        parsed_query = {
            "weather": "rainy",
            "time": "night",
            "objects": ["pedestrian"],
            "location": "intersection",
        }

        with patch.object(app, "query_scene_tokens", return_value=["scene-token"]), patch.object(
            app, "get_local_candidate_scene_tokens", return_value=[]
        ):
            tokens, status, should_stop = app.get_candidate_scene_tokens(parsed_query)

        self.assertEqual(tokens, [])
        self.assertIn("Neo4j", status)
        self.assertTrue(status)
        self.assertTrue(should_stop)
        self.assertNotIn("回退到全库搜索", status)

    def test_get_candidate_scene_tokens_reports_no_night_scene_in_current_like_subset(self):
        parsed_query = app.parse_query("夜间")
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

        self.assertEqual(parsed_query["time"], "night")
        self.assertEqual(tokens, [])
        self.assertTrue(should_stop)
        self.assertIn("85-scene", status)
        self.assertIn("night", status)
        self.assertNotIn("回退到全库搜索", status)

    def test_retrieve_images_skips_vector_search_when_structured_filters_have_no_candidates(self):
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
        ), patch.object(app, "encode_text_query") as encode_mock, patch.object(app, "search_frame_hits") as search_mock:
            image_results, model_name, parsed_query, kg_status = app.retrieve_images("夜间")

        self.assertEqual(parsed_query["time"], "night")
        self.assertEqual(image_results, [])
        self.assertEqual(model_name, app.KG_ZERO_CANDIDATE_MODEL_NAME)
        self.assertIn("85-scene", kg_status)
        self.assertIn("night", kg_status)
        encode_mock.assert_not_called()
        search_mock.assert_not_called()

    def test_retrieve_videos_skips_vector_search_when_structured_filters_have_no_candidates(self):
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
        ), patch.object(app, "encode_text_query") as encode_mock, patch.object(app, "search_frame_hits") as search_mock:
            video_results, model_name, parsed_query, kg_status = app.retrieve_videos("夜间")

        self.assertEqual(parsed_query["time"], "night")
        self.assertEqual(video_results, [])
        self.assertEqual(model_name, app.KG_ZERO_CANDIDATE_MODEL_NAME)
        self.assertIn("85-scene", kg_status)
        self.assertIn("night", kg_status)
        encode_mock.assert_not_called()
        search_mock.assert_not_called()

    def test_search_frame_hits_falls_back_to_local_scene_filter_when_filtered_search_is_empty(self):
        fake_collection = _FakeCollection()
        query_vector = np.zeros(4, dtype=np.float32)

        with patch.object(app, "get_live_collection", return_value=fake_collection), patch.object(
            app, "has_field", return_value=True
        ), patch.object(app, "get_search_output_fields", return_value=["scene_token", "frame_path"]), patch.object(
            app, "enrich_hit_record", side_effect=lambda record: record
        ):
            hits = app.search_frame_hits(query_vector, 1, ["scene-a"])

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["scene_token"], "scene-a")
        self.assertEqual(len(fake_collection.search_calls), 2)
        self.assertIn('scene_token in ["scene-a"]', fake_collection.search_calls[0]["expr"])
        self.assertEqual(fake_collection.search_calls[1]["expr"], f"id >= {app.IMAGE_ID_MIN}")

    def test_search_frame_hits_loads_collection_once_per_live_collection(self):
        fake_collection = _FakeCollection()
        query_vector = np.zeros(4, dtype=np.float32)

        with patch.object(app, "get_live_collection", return_value=fake_collection), patch.object(
            app, "has_field", return_value=False
        ), patch.object(app, "get_search_output_fields", return_value=["scene_token", "frame_path"]), patch.object(
            app, "enrich_hit_record", side_effect=lambda record: record
        ):
            app.search_frame_hits(query_vector, 1, [])
            app.search_frame_hits(query_vector, 1, [])

        self.assertEqual(fake_collection.load_calls, 1)

    def test_enrich_hit_record_reuses_cached_metadata_for_same_frame(self):
        record = {"id": 202, "score": 0.91, "frame_path": "samples/CAM_FRONT/match.jpg"}
        sample_data = {
            "sample_token": "sample-token",
            "scene_token": "scene-a",
            "camera": "CAM_FRONT",
            "filename": "samples/CAM_FRONT/match.jpg",
            "sample_data_token": "sample-data-token",
        }
        resolved_path = app.NUSCENES_ROOT / "samples" / "CAM_FRONT" / "match.jpg"

        with patch.object(app, "resolve_frame_path", return_value=resolved_path) as resolve_mock, patch.object(
            app, "get_sample_data_for_frame", return_value=sample_data
        ) as sample_mock:
            first = app.enrich_hit_record(dict(record))
            second = app.enrich_hit_record(dict(record))

        self.assertEqual(first["scene_token"], "scene-a")
        self.assertEqual(second["sample_data_token"], "sample-data-token")
        self.assertEqual(resolve_mock.call_count, 1)
        self.assertEqual(sample_mock.call_count, 1)

    def test_write_video_clip_reuses_cached_clip_for_same_sequence(self):
        frame_paths = [
            app.NUSCENES_ROOT / "samples" / "CAM_FRONT" / "frame_0001.jpg",
            app.NUSCENES_ROOT / "samples" / "CAM_FRONT" / "frame_0002.jpg",
        ]
        anchor_record = {"scene_name": "scene-a", "scene_token": "scene-a", "camera": "CAM_FRONT"}
        cache_dir = Path(app.__file__).resolve().parent / ".tmp_test_video_cache"

        def fake_render(output_path, *_args):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"video")
            return output_path

        shutil.rmtree(cache_dir, ignore_errors=True)
        with patch.object(app, "GENERATED_VIDEO_DIR", cache_dir), patch.object(
            app, "render_video_candidate", side_effect=fake_render
        ) as render_mock, patch.object(app, "is_browser_playable_clip", side_effect=lambda path: path.exists()):
            first = app.write_video_clip(anchor_record, frame_paths)
            second = app.write_video_clip(anchor_record, frame_paths)

        self.assertEqual(first, second)
        self.assertEqual(render_mock.call_count, 1)
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_build_query_highlight_html_renders_marker_markup(self):
        parsed = app.parse_query("rainy intersection with cars at night")
        html = app.build_query_highlight_html(parsed)

        self.assertIn("query-highlight-block", html)
        self.assertIn("query-marker marker-weather", html)
        self.assertIn("query-marker marker-time", html)
        self.assertIn("query-marker marker-location", html)
        self.assertIn("query-marker marker-object", html)
        self.assertIn("query-highlight-chip", html)
        self.assertIn("rainy", html)
        self.assertIn("cars", html)

    def test_build_explanation_html_renders_top_runtime_bar_and_two_focus_cards(self):
        parsed_query = {
            "weather": "rainy",
            "time": "night",
            "objects": ["pedestrian"],
            "location": "intersection",
            "raw_text": "rainy night intersection with pedestrians",
            "highlight_matches": [
                {"start": 0, "end": 5, "text": "rainy", "category": "weather", "label": "rainy"},
                {"start": 6, "end": 11, "text": "night", "category": "time", "label": "night"},
                {"start": 12, "end": 24, "text": "intersection", "category": "location", "label": "intersection"},
                {"start": 30, "end": 41, "text": "pedestrians", "category": "object", "label": "pedestrian"},
            ],
        }
        html = app.build_explanation_html(
            parsed_query,
            "Neo4j filtered 1 scene | 耗时：NLP=12ms, 知识图谱=34ms, Milvus=56ms",
            "Chinese-CLIP",
            "text2image",
            3,
        )

        self.assertIn("query-highlight-block", html)
        self.assertIn("detail-panel", html)
        self.assertIn("runtime-strip", html)
        self.assertIn("runtime-total", html)
        self.assertIn("runtime-breakdown", html)
        self.assertIn("focus-card-grid", html)
        self.assertIn("focus-card nlp-focus-card", html)
        self.assertIn("focus-card kg-focus-card", html)
        self.assertIn("focus-card-body", html)
        self.assertIn("module-inline-details", html)
        self.assertIn("kg-path-block", html)
        self.assertIn("path-segment path-segment-relation", html)
        self.assertIn("总耗时 0.10s", html)
        self.assertIn("NLP 12ms", html)
        self.assertIn("query-marker marker-weather", html)
        self.assertNotIn("kg-runtime-meta", html)
        self.assertNotIn("path-summary-chip", html)
        self.assertNotIn("Chinese-CLIP", html)

    def test_custom_css_keeps_five_column_images_with_custom_preview_button(self):
        self.assertNotIn("button:hover, .gr-button:hover", app.custom_css)
        self.assertIn("#search-btn:hover, #clear-btn:hover", app.custom_css)
        self.assertIn("grid-template-columns: repeat(5, minmax(0, 1fr));", app.custom_css)
        self.assertIn("body.lightbox-open", app.custom_css)
        self.assertIn("background: rgba(248, 250, 252", app.custom_css)
        self.assertIn("100dvh", app.custom_css)
        self.assertIn(".query-marker", app.custom_css)
        self.assertIn(".marker-time", app.custom_css)
        self.assertIn(".query-highlight-chip", app.custom_css)
        self.assertIn(".module-inline-details", app.custom_css)
        self.assertIn(".kg-path-block", app.custom_css)
        self.assertIn(".path-segment-relation", app.custom_css)
        self.assertIn(".focus-card-grid", app.custom_css)
        self.assertIn(".focus-card-body", app.custom_css)
        self.assertIn(".runtime-strip", app.custom_css)
        self.assertIn(".runtime-total", app.custom_css)
        self.assertIn(".runtime-breakdown", app.custom_css)
        self.assertNotIn("max-height: 132px;", app.custom_css)
        self.assertNotIn("overflow: auto;", app.custom_css)
        source = Path(app.__file__).read_text(encoding="utf-8")
        self.assertIn("gr.Image(", source)
        self.assertIn('gr.Button("放大查看"', source)
        self.assertIn("def open_image_preview", source)
        self.assertIn("document.body.classList.add('lightbox-open')", source)
        self.assertIn("document.body.classList.remove('lightbox-open')", source)
        self.assertNotIn("status_panel = gr.HTML", source)
        self.assertNotIn("gr.Gallery(", source)
        self.assertEqual(app.MODE_LABELS["text2image"], "搜索图片")
        self.assertEqual(app.MODE_LABELS["text2video"], "搜索视频片段")
        self.assertIn("show_fullscreen_button=False", source)


if __name__ == "__main__":
    unittest.main()

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

    def test_build_explanation_html_renders_query_and_kg_sections(self):
        html = app.build_explanation_html(
            {"weather": "rainy", "time": "night", "objects": ["pedestrian"], "location": "intersection"},
            "Neo4j filtered 1 scene",
            "Chinese-CLIP",
            "text2image",
            3,
        )

        self.assertIn("<details", html)
        self.assertIn("detail-panel", html)
        self.assertIn("module-card", html)
        self.assertIn("rainy", html)
        self.assertIn("night", html)
        self.assertIn("pedestrian", html)
        self.assertIn("Neo4j", html)
        self.assertNotIn("Chinese-CLIP", html)

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

        expected_output_count = app.IMAGE_RESULT_COUNT * 2 + app.VIDEO_RESULT_COUNT * 2 + 5
        self.assertEqual(len(output), expected_output_count)
        self.assertIsNotNone(output[0])
        self.assertEqual(output[app.IMAGE_RESULT_COUNT], "caption")
        self.assertIsNone(output[-5])
        self.assertEqual(output[-4], "")
        self.assertIsInstance(output[-3], dict)
        self.assertFalse(output[-3]["visible"])
        self.assertIn("status-card success", output[-2])
        self.assertIn("English-CLIP", output[-2])
        self.assertIn("Neo4j filtered 1 scene", output[-2])
        self.assertIn("detail-panel", output[-1])
        self.assertIn("module-card", output[-1])
        self.assertNotIn("Chinese-CLIP", output[-1])

    def test_custom_css_keeps_five_column_images_with_custom_preview_button(self):
        self.assertNotIn("button:hover, .gr-button:hover", app.custom_css)
        self.assertIn("#search-btn:hover, #clear-btn:hover", app.custom_css)
        self.assertIn("grid-template-columns: repeat(5, minmax(0, 1fr));", app.custom_css)
        self.assertIn("body.lightbox-open", app.custom_css)
        self.assertIn("background: rgba(248, 250, 252", app.custom_css)
        self.assertIn("100dvh", app.custom_css)
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
            tokens, status = app.get_candidate_scene_tokens(parsed_query)

        self.assertEqual(tokens, ["scene-a"])
        self.assertIn("Neo4j", status)
        self.assertTrue(status)

    def test_get_candidate_scene_tokens_falls_back_to_full_collection_when_no_valid_tokens_exist(self):
        parsed_query = {
            "weather": "rainy",
            "time": "night",
            "objects": ["pedestrian"],
            "location": "intersection",
        }

        with patch.object(app, "query_scene_tokens", return_value=["scene-token"]), patch.object(
            app, "get_local_candidate_scene_tokens", return_value=[]
        ):
            tokens, status = app.get_candidate_scene_tokens(parsed_query)

        self.assertEqual(tokens, [])
        self.assertIn("Neo4j", status)
        self.assertTrue(status)

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


if __name__ == "__main__":
    unittest.main()

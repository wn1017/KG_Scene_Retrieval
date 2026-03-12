import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app


class AppInterfaceTests(unittest.TestCase):
    def test_dynamic_retrieve_accepts_image_retrieval_metadata_payload(self):
        fake_image = Image.new("RGB", (8, 8), color="white")
        fake_result = (
            [(fake_image, "caption")],
            "English-CLIP",
            {"weather": "rainy", "objects": ["pedestrian"]},
            "Neo4j filtered 1 scene",
        )

        with patch.object(app, "retrieve_images", return_value=fake_result):
            output = app.dynamic_retrieve("rainy night with pedestrians", "text2image")

        expected_output_count = app.IMAGE_RESULT_COUNT * 2 + app.VIDEO_RESULT_COUNT * 2 + 1
        self.assertEqual(len(output), expected_output_count)
        self.assertIsNotNone(output[0])
        self.assertIn("status-card success", output[-1])
        self.assertIn("English-CLIP", output[-1])
        self.assertIn("Neo4j filtered 1 scene", output[-1])

    def test_write_video_clip_generates_browser_playable_media(self):
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


if __name__ == "__main__":
    unittest.main()

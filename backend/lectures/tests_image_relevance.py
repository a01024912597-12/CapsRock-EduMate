"""이미지 관련성 검증 회귀 테스트. 외부 API를 호출하지 않는다."""

import os
import tempfile
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from .services.image_relevance_service import (
    _extract_selected_ids,
    _image_quality_is_usable,
    select_relevant_timeline_frames,
)
from .services.summary_service import inject_core_timeline_images


class ImageRelevanceTests(SimpleTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.video = os.path.join(self.folder.name, "video.mp4")
        # 프레임 수집은 모킹하므로 유효한 mp4 파일일 필요가 없다.
        with open(self.video, "wb") as output:
            output.write(b"video-placeholder")
        self.override = override_settings(MEDIA_ROOT=self.folder.name)
        self.override.enable()
        self.addCleanup(self.override.disable)

    def _candidate(self):
        return {
            "id": "0_0", "concept_index": 0, "keyword": "난징조약",
            "target_seconds": 310, "capture_seconds": 315,
            "context": "난징조약의 체결", "jpeg": b"fake-jpeg-data",
            "fingerprint": 0x0F0F0F0F0F0F0F0F,
        }

    def test_unknown_ai_candidate_is_ignored(self):
        picked = _extract_selected_ids('{"selected_ids":["no_such_frame"]}', [self._candidate()])
        self.assertEqual(picked, [])

    def test_api_declines_all_frames(self):
        with patch("lectures.services.image_relevance_service._collect_candidates", return_value=[self._candidate()]):
            with patch("lectures.services.image_relevance_service._review_with_openai", return_value='{"selected_ids":[]}'):
                frames = select_relevant_timeline_frames(
                    self.video, 100, [{"keyword": "난징조약", "seconds": 310}],
                    "3. 꼭 알아야 할 필수 개념", summary_api="gpt",
                )
        self.assertEqual(frames, [])
        self.assertFalse(os.path.exists(os.path.join(self.folder.name, "lecture_frames", "100")))

    def test_only_approved_frames_are_saved(self):
        with patch("lectures.services.image_relevance_service._collect_candidates", return_value=[self._candidate()]):
            with patch("lectures.services.image_relevance_service._review_with_openai", return_value='{"selected_ids":["0_0"]}'):
                frames = select_relevant_timeline_frames(
                    self.video, 100, [{"keyword": "난징조약", "seconds": 310}],
                    "3. 꼭 알아야 할 필수 개념", summary_api="gpt",
                )
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0]["validated"])
        self.assertEqual(frames[0]["target_seconds"], 310)
        self.assertEqual(frames[0]["time_str"], "05:15")
        with open(frames[0]["filepath"], "rb") as saved:
            self.assertEqual(saved.read(), b"fake-jpeg-data")

    def test_image_insertion_requires_exact_validated_concept(self):
        path = os.path.join(self.folder.name, "verified.jpg")
        with open(path, "wb") as output:
            output.write(b"fake-image-data")
        frames = [{
            "filepath": path, "validated": True, "target_seconds": 310,
            "seconds": 315, "time_str": "05:15", "keyword": "난징조약",
        }]
        summary = (
            "3. 꼭 알아야 할 필수 개념\n"
            "- **난징조약 (05:10)**\n"
            "- **기술혁명 (05:30)**\n"
            "4. 핵심 내용\n"
        )
        result = inject_core_timeline_images(summary, frames, lecture_id=100)
        self.assertEqual(result.count('class="timeline-image-wrapper"'), 1)
        self.assertIn('05:15', result)
        self.assertIn('난징조약', result)

    def test_unvalidated_image_is_never_inserted(self):
        path = os.path.join(self.folder.name, "old.jpg")
        with open(path, "wb") as output:
            output.write(b"fake-image-data")
        summary = "3. 꼭 알아야 할 필수 개념\n- **난징조약 (05:10)**\n4. 핵심 내용"
        frame = {"filepath": path, "target_seconds": 310, "keyword": "난징조약"}
        result = inject_core_timeline_images(summary, [frame], lecture_id=100)
        self.assertNotIn('class="timeline-image-wrapper"', result)

    def test_blank_image_is_rejected(self):
        import numpy as np
        self.assertFalse(_image_quality_is_usable(np.zeros((360, 640, 3), dtype=np.uint8)))

    def test_save_high_quality_display_jpeg_instead_of_review_thumbnail(self):
        candidate = self._candidate()
        candidate["display_jpeg"] = b"high-quality-jpeg"
        candidate["display_size"] = (1920, 1080)
        with patch("lectures.services.image_relevance_service._collect_candidates", return_value=[candidate]):
            with patch("lectures.services.image_relevance_service._review_with_openai", return_value='{"selected_ids":["0_0"]}'):
                frames = select_relevant_timeline_frames(
                    self.video, 100, [{"keyword": "난징조약", "seconds": 310}],
                    "3. 꼭 알아야 할 필수 개념", summary_api="gpt",
                )
        with open(frames[0]["filepath"], "rb") as saved:
            self.assertEqual(saved.read(), b"high-quality-jpeg")

    def test_keep_original_aspect_ratio_without_upscaling(self):
        import numpy as np
        from .services.image_relevance_service import _fit_frame_max_size
        big = np.zeros((2160, 3840, 3), dtype=np.uint8)
        small = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(_fit_frame_max_size(big, (1920, 1080)).shape, (1080, 1920, 3))
        self.assertEqual(_fit_frame_max_size(small, (1920, 1080)).shape, (240, 320, 3))

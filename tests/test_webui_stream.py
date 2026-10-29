import unittest
from unittest.mock import patch

import numpy as np
import torch

import webui


class FakeCosyVoice:
    sample_rate = 24000

    class frontend:
        @staticmethod
        def text_normalize(text, split=True, text_frontend=True):
            return [text]

    @staticmethod
    def inference_sft(tts_text, speaker, stream=False, speed=1.0):
        for sample in (1, 2, 3):
            yield {'tts_speech': torch.tensor([[sample]], dtype=torch.float32)}


class WebUIStreamTests(unittest.TestCase):
    def test_stream_preview_contains_all_previous_audio_chunks(self):
        saved = []

        def capture_audio(audio, sample_rate):
            saved.append(np.asarray(audio).copy())
            return 'audio-{}.wav'.format(len(saved))

        with patch.object(webui, 'cosyvoice', FakeCosyVoice(), create=True), \
             patch.object(webui, 'MODEL_IS_COSYVOICE3', False), \
             patch.object(webui, 'save_audio', side_effect=capture_audio), \
             patch.object(webui, 'ensure_notification_audio', return_value='done.wav'), \
             patch.object(webui, 'set_all_random_seed'), \
             patch.object(webui.gr, 'Info'):
            updates = list(webui.generate_audio('测试。', '预训练音色', 'speaker', '', None, None, '', 0, True, 1.0))

        self.assertEqual([audio.tolist() for audio in saved], [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(len(updates), 5)
        self.assertEqual(updates[-1][2], 'done.wav')


if __name__ == '__main__':
    unittest.main()

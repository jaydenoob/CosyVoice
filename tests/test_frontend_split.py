import unittest

from cosyvoice.utils.frontend_utils import split_paragraph


class SplitParagraphTests(unittest.TestCase):
    def test_long_chinese_sentence_is_split_without_losing_text(self):
        text = '这是一句没有标点的较长中文文本' * 10
        segments = split_paragraph(text, lambda value: list(value), lang='zh')
        self.assertEqual(''.join(segments), text + '。')
        self.assertTrue(all(len(segment) <= 80 for segment in segments))

    def test_short_tail_does_not_exceed_segment_limit_when_merged(self):
        text = '甲' * 75 + '。' + '乙' * 15 + '。'
        segments = split_paragraph(text, lambda value: list(value), lang='zh')
        self.assertEqual(''.join(segments), text)
        self.assertTrue(all(len(segment) <= 80 for segment in segments))


if __name__ == '__main__':
    unittest.main()

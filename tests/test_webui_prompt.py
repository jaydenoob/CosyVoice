import unittest

from webui import format_cosyvoice3_instruction, format_cosyvoice3_prompt_text


class CosyVoice3PromptTests(unittest.TestCase):
    def test_reference_transcript_follows_delimiter(self):
        self.assertEqual(
            format_cosyvoice3_prompt_text('参考音频说的话。'),
            'You are a helpful assistant.<|endofprompt|>参考音频说的话。',
        )

    def test_dialect_instruction_precedes_delimiter(self):
        self.assertEqual(
            format_cosyvoice3_instruction('请用四川话说这句话'),
            'You are a helpful assistant. 请用四川话说这句话<|endofprompt|>',
        )

    def test_complete_prompts_are_preserved(self):
        reference = 'You are a helpful assistant.<|endofprompt|>参考音频说的话。'
        instruction = 'You are a helpful assistant. 请用四川话说这句话<|endofprompt|>'
        self.assertEqual(format_cosyvoice3_prompt_text(reference), reference)
        self.assertEqual(format_cosyvoice3_instruction(instruction), instruction)


if __name__ == '__main__':
    unittest.main()

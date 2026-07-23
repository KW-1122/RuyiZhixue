import unittest

from ruyitutor.response_cleaning import strip_thinking


class ResponseCleaningTests(unittest.TestCase):
    def test_strip_complete_think_block(self):
        answer = strip_thinking("<think>hidden reasoning</think>\n正式回答")
        self.assertEqual(answer, "正式回答")

    def test_strip_unmatched_think_tags(self):
        answer = strip_thinking("<think>hidden\n正式回答</think>")
        self.assertEqual(answer, "")

    def test_plain_answer_is_preserved(self):
        self.assertEqual(strip_thinking("正式回答"), "正式回答")


if __name__ == "__main__":
    unittest.main()

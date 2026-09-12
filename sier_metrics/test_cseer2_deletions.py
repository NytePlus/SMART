import unittest

from _text_units import build_hyp_error_flags, error_chunk_count_in_hypothesis


class Cseer2DeletionAnchorsTest(unittest.TestCase):
    def test_internal_deletions_anchor_hypothesis_chunk(self) -> None:
        flags = build_hyp_error_flags("PAINTING", "====+=+===")

        self.assertEqual([index for index, value in enumerate(flags) if value], [3, 4])
        self.assertEqual(error_chunk_count_in_hypothesis("PAINTING", "====+=+==="), 1)

    def test_trailing_deletions_anchor_last_hypothesis_character(self) -> None:
        self.assertEqual(error_chunk_count_in_hypothesis("GIANT", "=====++"), 1)

    def test_leading_deletion_anchor_is_not_cleared_by_following_match(self) -> None:
        self.assertEqual(build_hyp_error_flags("WORD", "+===="), [True, False, False, False])
        self.assertEqual(error_chunk_count_in_hypothesis("WORD", "+===="), 1)

    def test_exact_match_still_has_no_error_chunk(self) -> None:
        self.assertEqual(error_chunk_count_in_hypothesis("WORD", "===="), 0)

    def test_punctuation_only_hypothesis_preserves_deletion(self) -> None:
        self.assertEqual(error_chunk_count_in_hypothesis("—", "+++$"), 1)


if __name__ == "__main__":
    unittest.main()

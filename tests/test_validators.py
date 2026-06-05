from __future__ import annotations

import unittest

from local_micro_agent.validators import JsonValidationError, parse_json_object


class ValidatorTests(unittest.TestCase):
    def test_parse_json_object_wraps_decode_errors(self) -> None:
        with self.assertRaises(JsonValidationError):
            parse_json_object("{changes: []}")

    def test_parse_json_object_extracts_object_from_prose(self) -> None:
        self.assertEqual(parse_json_object("Here:\n{\"ok\": true}\n"), {"ok": True})


if __name__ == "__main__":
    unittest.main()

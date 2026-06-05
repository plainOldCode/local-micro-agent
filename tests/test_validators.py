from __future__ import annotations

import unittest

from local_micro_agent.validators import JsonValidationError, parse_json_object, parse_xml_candidates


class ValidatorTests(unittest.TestCase):
    def test_parse_json_object_wraps_decode_errors(self) -> None:
        with self.assertRaises(JsonValidationError):
            parse_json_object("{changes: []}")

    def test_parse_json_object_extracts_object_from_prose(self) -> None:
        self.assertEqual(parse_json_object("Here:\n{\"ok\": true}\n"), {"ok": True})

    def test_parse_xml_candidates_extracts_raw_code_blocks(self) -> None:
        data = parse_xml_candidates(
            """
<candidates>
<candidate id="fast-path">
<reason>hoist independent address calculations</reason>
<change>
<path>perf_takehome.py</path>
<search>
body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
</search>
<replace>
body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
</replace>
</change>
</candidate>
</candidates>
"""
        )

        self.assertEqual(data["candidates"][0]["id"], "fast-path")
        change = data["candidates"][0]["changes"][0]
        self.assertEqual(change["path"], "perf_takehome.py")
        self.assertIn('self.scratch["inp_indices_p"]', change["replacement"])


if __name__ == "__main__":
    unittest.main()

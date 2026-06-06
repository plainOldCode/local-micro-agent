from __future__ import annotations

import unittest

from local_micro_agent.validators import (
    JsonValidationError,
    XmlValidationError,
    parse_json_object,
    parse_xml_candidates,
)


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
<strategy_axis>instruction_scheduling</strategy_axis>
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
        self.assertEqual(data["candidates"][0]["strategy_axis"], "instruction_scheduling")
        change = data["candidates"][0]["changes"][0]
        self.assertEqual(change["path"], "perf_takehome.py")
        self.assertIn('self.scratch["inp_indices_p"]', change["replacement"])
        self.assertIn('body.append(("debug"', change["target"])

    def test_parse_xml_candidates_tolerates_raw_less_than_operator(self) -> None:
        data = parse_xml_candidates(
            """
<candidates>
<candidate id="1">
<reason>small local edit</reason>
<change>
<path>perf_takehome.py</path>
<search>
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
</search>
<replace>
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("debug", ("compare", tmp_idx, (round, i, "lt_check"))))
</replace>
</change>
</candidate>
</candidates>
"""
        )

        change = data["candidates"][0]["changes"][0]
        self.assertIn('("<", tmp1', change["target"])
        self.assertTrue(change["target"].startswith("                body.append"))

    def test_xml_validation_error_uses_json_error_path(self) -> None:
        self.assertTrue(issubclass(XmlValidationError, JsonValidationError))


if __name__ == "__main__":
    unittest.main()

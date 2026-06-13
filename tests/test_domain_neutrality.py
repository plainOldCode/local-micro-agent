from pathlib import Path
import re
import unittest


class DomainNeutralityTests(unittest.TestCase):
    def test_runtime_sources_do_not_hardcode_takehome_domain_terms(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        scanned_roots = [
            repo / "src",
            repo / "config",
            repo / "README.md",
        ]
        banned = re.compile(
            r"\b("
            r"takehome|perf_takehome|KernelBuilder|build_kernel|VLIW|SIMD|"
            r"scratch_const|issue_slot_pressure|instruction_scheduling|hash_reorder"
            r")\b"
        )
        hits: list[str] = []
        for root in scanned_roots:
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in paths:
                if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg"}:
                    continue
                text = path.read_text(errors="replace")
                for line_no, line in enumerate(text.splitlines(), 1):
                    if banned.search(line):
                        hits.append(f"{path.relative_to(repo)}:{line_no}: {line.strip()}")

        self.assertEqual([], hits)


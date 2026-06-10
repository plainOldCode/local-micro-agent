"""Prompt context assembly: project/external context, source excerpts, slicing.

Extracted from orchestrator.py; mixed into MicroAgent.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..prompts import semantic_analysis_prompt
from ..state import ExternalContext


class PromptContextMixin:
    async def _maybe_refresh_semantic_analysis(self) -> None:
        workflow = self.config.get("workflow", {})
        path = self._workflow_artifact_path(
            "semantic_analysis_path", ".local_micro_agent/semantic_analysis.md"
        )
        if path.exists():
            text = self._curate_semantic_analysis(
                path.read_text(errors="replace"),
                int(workflow.get("semantic_analysis_char_limit", 8000) or 8000),
            )
            if text:
                self.state.scratch["semantic_analysis"] = text
                self.state.notes.append(f"Loaded semantic analysis: {path}")
        if not workflow.get("semantic_analysis_after_read"):
            return
        focus = self._focused_read_model_context(
            str(workflow.get("semantic_analysis_focus", ""))
        )
        role = str(workflow.get("semantic_analysis_model_role", "planner"))
        try:
            output = await self._model_chat(
                role,
                semantic_analysis_prompt(self.state, focus=focus),
                call_site="semantic_analysis",
            )
        except Exception as exc:
            self.state.notes.append(
                f"Semantic analysis model call failed: {type(exc).__name__}: {exc}"
            )
            return
        analysis = output.strip()
        if not analysis:
            return
        limit = int(workflow.get("semantic_analysis_char_limit", 8000) or 8000)
        analysis = self._slice_text(analysis, limit)
        curated = self._curate_semantic_analysis(analysis, limit)
        if not curated:
            self.state.notes.append("Semantic analysis discarded: no code-usable facts")
            return
        self.state.scratch["semantic_analysis"] = curated
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis + "\n")
        curated_path = self._workflow_artifact_path(
            "semantic_analysis_curated_path",
            ".local_micro_agent/semantic_analysis.curated.md",
        )
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        curated_path.write_text(curated + "\n")
        self.state.notes.append(f"Persisted semantic analysis: {path}")

    def _focused_read_model_context(self, configured_focus: str = "") -> str:
        parts = [configured_focus.strip()] if configured_focus.strip() else []
        focused = self.state.scratch.get("focused_read_context")
        if isinstance(focused, str) and focused.strip():
            parts.append(f"Focused read context:\n{focused}")
        return "\n\n".join(parts)

    def _filter_read_files(self, files: list[str]) -> list[str]:
        external_paths = self._external_context_path_keys()
        filtered: list[str] = []
        seen: set[str] = set()
        for raw_path in files:
            rel_path = str(raw_path).strip()
            if not rel_path:
                continue
            path_key = self._repo_path_key(rel_path)
            if path_key in external_paths:
                self.state.notes.append(
                    f"Skipped advisory external context as source file: {rel_path}"
                )
                continue
            if path_key in seen:
                continue
            seen.add(path_key)
            filtered.append(rel_path)
        return filtered

    def _external_context_path_keys(self) -> set[str]:
        keys: set[str] = set()
        for spec in self._external_context_specs():
            for raw_path in (spec.get("path"), spec.get("source")):
                if not raw_path:
                    continue
                keys.add(self._repo_path_key(str(raw_path)))
        return {key for key in keys if key}

    def _repo_path_key(self, path: str) -> str:
        candidate = Path(path)
        repo_root = self.state.repo_root.resolve(strict=False)
        resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (repo_root / candidate).resolve(strict=False)
        )
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            return str(resolved)

    def _workflow_artifact_path(self, key: str, default: str) -> Path:
        raw = self.config.get("workflow", {}).get(key, default)
        path = Path(str(raw))
        if path.is_absolute():
            return path
        return self.state.repo_root / path

    async def _load_external_contexts(self) -> None:
        specs = self._external_context_specs()
        if not specs:
            self.state.external_context = []
            return
        workflow = self.config.get("workflow", {})
        total_limit = int(workflow.get("external_context_char_limit", 12000) or 0)
        item_limit = int(workflow.get("external_context_item_char_limit", 6000) or 0)
        if total_limit <= 0 or item_limit <= 0:
            self.state.external_context = []
            return
        contexts: list[ExternalContext] = []
        remaining = total_limit
        for spec in specs:
            raw_path = str(spec.get("path") or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            abs_path = path if path.is_absolute() else self.state.repo_root / path
            source = str(spec.get("source") or raw_path)
            if remaining <= 0:
                break
            try:
                content = await self.mcp.read_file(str(abs_path))
            except FileNotFoundError:
                self.state.notes.append(f"External context file not found: {source}")
                continue
            sha = hashlib.sha256(content.encode()).hexdigest()
            budget = min(item_limit, remaining)
            sliced = self._slice_text(content, budget)
            remaining -= len(sliced)
            contexts.append(
                ExternalContext(
                    kind=str(spec.get("kind") or "hint"),
                    source=source,
                    title=str(spec.get("title") or self._external_context_title(content, source)),
                    content=sliced,
                    sha256=sha,
                    trust=str(spec.get("trust") or "advisory"),
                    fetched_at=spec.get("fetched_at"),
                )
            )
        self.state.external_context = contexts
        sources_key = tuple((item.source, item.sha256) for item in contexts)
        if contexts and self.state.scratch.get("external_context_sources") != sources_key:
            self.state.notes.append(
                "Loaded external context: " + ", ".join(item.source for item in contexts)
            )
            self.state.scratch["external_context_sources"] = sources_key

    def _external_context_specs(self) -> list[dict[str, Any]]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("external_context_paths")
        if configured in (None, "", []):
            return []
        if not isinstance(configured, list):
            configured = [configured]
        specs: list[dict[str, Any]] = []
        for item in configured:
            if isinstance(item, dict):
                raw_path = item.get("path") or item.get("source")
                if not raw_path:
                    continue
                spec = {
                    "path": str(raw_path),
                    "source": str(item.get("source") or raw_path),
                    "title": str(item.get("title") or ""),
                    "kind": str(item.get("kind") or "hint"),
                    "trust": str(item.get("trust") or "advisory"),
                }
                fetched_at = item.get("fetched_at")
                if fetched_at:
                    spec["fetched_at"] = str(fetched_at)
                specs.append(spec)
            else:
                raw_path = str(item).strip()
                if raw_path:
                    specs.append(
                        {
                            "path": raw_path,
                            "source": raw_path,
                            "title": "",
                            "kind": "hint",
                            "trust": "advisory",
                        }
                    )
        return specs

    @staticmethod
    def _external_context_title(content: str, source: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
        return Path(source).name or source

    async def _load_project_context(self) -> str:
        files = self._project_context_files()
        if not files:
            return ""
        limit = int(self.config.get("workflow", {}).get("project_context_char_limit", 12000))
        blocks = []
        for rel_path in files:
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / rel_path))
            except FileNotFoundError:
                self.state.notes.append(f"Project context file not found: {rel_path}")
                continue
            blocks.append(f"### {rel_path}\n```text\n{self._slice_text(content, limit)}\n```")
        if blocks:
            self.state.notes.append(
                "Loaded project context: " + ", ".join(files)
            )
        return "\n\n".join(blocks)

    def _project_context_files(self) -> list[str]:
        workflow = self.config.get("workflow", {})
        configured = workflow.get("project_context_files")
        if isinstance(configured, list) and configured:
            return [str(path) for path in configured]
        files = []
        instruction_files = workflow.get("project_instruction_files")
        if isinstance(instruction_files, list) and instruction_files:
            files.extend(str(path) for path in instruction_files)
        else:
            files.extend(
                name
                for name in ("AGENTS.md", "CLAUDE.md", "INSTRUCTIONS.md")
                if (self.state.repo_root / name).exists()
            )
        if workflow.get("readme_first", True) is False:
            return self._unique_existing_paths(files)
        for name in ("README.md", "Readme.md", "readme.md", "README", "README.txt"):
            if (self.state.repo_root / name).exists():
                files.append(name)
                break
        return self._unique_existing_paths(files)

    def _unique_existing_paths(self, paths: list[str]) -> list[str]:
        unique = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if (self.state.repo_root / path).exists():
                unique.append(path)
        return unique

    def _workflow_plan_context(self) -> str:
        workflow = self.config.get("workflow", {})
        keys = (
            "writable_files",
            "test_commands",
            "metric_regex",
            "metric_goal",
            "baseline_metric",
            "accept_if_improved",
            "require_metric",
        )
        summary = {key: workflow[key] for key in keys if key in workflow and workflow[key] not in (None, [], "")}
        if not summary:
            return ""
        return "### Workflow constraints\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```"

    @staticmethod
    def _slice_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + "\n[...truncated...]\n" + text[-tail:]

    @staticmethod
    def _line_numbered_text(text: str, start_line: int = 1) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        width = len(str(start_line + len(lines) - 1))
        return "\n".join(
            f"{line_no:>{width}}: {line}"
            for line_no, line in enumerate(lines, start=start_line)
        )

    @staticmethod
    def _anchor_tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
            if len(token) >= 3
        }

    def _line_numbered_context(self, text: str, limit: int) -> str:
        sliced = self._slice_text(text, limit)
        return self._line_numbered_text(sliced)

    def _best_anchor_excerpt(
        self,
        content: str,
        anchor: str,
        *,
        context_lines: int,
        limit: int,
    ) -> str:
        lines = content.splitlines()
        if not lines:
            return ""
        tokens = self._anchor_tokens(anchor)
        best_index = 0
        best_score = -1
        if tokens:
            for index, line in enumerate(lines):
                line_tokens = self._anchor_tokens(line)
                score = len(tokens & line_tokens)
                if score > best_score:
                    best_index = index
                    best_score = score
        start = max(0, best_index - context_lines)
        end = min(len(lines), best_index + context_lines + 1)
        excerpt = "\n".join(lines[start:end])
        if len(excerpt) > limit:
            excerpt = self._slice_text(excerpt, limit)
            start_line = 1
        else:
            start_line = start + 1
        return self._line_numbered_text(excerpt, start_line=start_line)

    async def _format_current_source_context(self) -> str:
        workflow = self.config.get("workflow", {})
        if not workflow.get("current_source_context_before_code", True):
            return ""
        paths = sorted(self._writable_files())
        if not paths:
            return ""
        limit = int(workflow.get("current_source_context_char_limit", 12000) or 12000)
        per_file_limit = max(1000, limit // max(len(paths), 1))
        blocks: list[str] = []
        for rel_path in paths:
            try:
                content = await self.mcp.read_file(str(self.state.repo_root / rel_path))
            except FileNotFoundError:
                blocks.append(f"### {rel_path}\n<missing>")
                continue
            content = self._context_for_file(rel_path, content, record_note=False)
            numbered = self._line_numbered_context(content, per_file_limit)
            blocks.append(f"### {rel_path}\n```text\n{numbered}\n```")
        return self._slice_text("\n\n".join(blocks), limit)

    @classmethod
    def _curate_semantic_analysis(cls, text: str, limit: int) -> str:
        """Keep only semantic context that is safe to feed into CODE attempts."""
        code_usable_sections = {
            "code-usable facts",
            "hazards and ordering constraints",
            "current task metric constraints",
            "safe implementation hooks",
            "execution model / data visibility",
            "invariants and public contracts",
            "risky transformations and required checks",
        }
        non_constraint_sections = {
            "background / non-constraints",
            "background",
            "non-constraints",
            "notes",
            "leaderboard",
        }
        background_patterns = (
            "best known",
            "leaderboard",
            "prior run",
            "previous run",
            "benchmark note",
            "outside this run",
        )
        lowered_all = text.lower()
        delayed_visibility = bool(
            re.search(r"\bread[s]?\b.*\b(before|previous|old)\b.*\bwrite", lowered_all)
            or re.search(r"\bwrite[s]?\b.*\b(end|after|later)\b", lowered_all)
        )
        lines: list[str] = []
        current_section: str | None = None
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            section_match = re.match(r"^#{1,6}\s*(.+?)\s*$", stripped)
            bullet_section_match = re.match(r"^[-*]\s*([A-Za-z][^:]{1,80}):\s*$", stripped)
            plain_section_match = (
                stripped.lower()
                if stripped.lower() in code_usable_sections | non_constraint_sections
                else None
            )
            if section_match:
                current_section = section_match.group(1).strip().lower()
            elif bullet_section_match:
                current_section = bullet_section_match.group(1).strip().lower()
            elif plain_section_match:
                current_section = plain_section_match
            section_key = (current_section or "").strip()
            if section_key in non_constraint_sections:
                continue
            lowered = stripped.lower()
            if any(pattern in lowered for pattern in background_patterns):
                continue
            if delayed_visibility and re.search(
                r"\b(no|not|without)\b.*\b(data\s+dependency|dependency|hazard)",
                lowered,
            ):
                continue
            if section_key and section_key not in code_usable_sections:
                if not stripped.startswith(("-", "*")):
                    continue
            if stripped:
                lines.append(raw_line.rstrip())

        curated = "\n".join(lines).strip()
        if delayed_visibility:
            warning = (
                "Controller validation: delayed write visibility or read-before-write "
                "ordering is an execution hazard. Do not move producer/consumer work "
                "into the same logical step unless the source proves the read still "
                "observes the required value."
            )
            if warning not in curated:
                curated = f"{curated}\n\n{warning}" if curated else warning
        if not curated:
            return ""
        return cls._slice_text(curated, limit)

    def _context_for_file(self, rel_path: str, content: str, record_note: bool = True) -> str:
        symbols_by_path = self.config.get("workflow", {}).get("context_symbols")
        if not isinstance(symbols_by_path, dict):
            return content
        symbols = symbols_by_path.get(rel_path)
        if not symbols:
            return content
        if not isinstance(symbols, list):
            if record_note:
                self.state.notes.append(f"Ignored non-list context_symbols for {rel_path}")
            return content
        excerpt = self._extract_python_symbols(content, [str(symbol) for symbol in symbols])
        if not excerpt:
            if record_note:
                self.state.notes.append(f"No requested context symbols found in {rel_path}")
            return content
        if record_note:
            self.state.notes.append(
                f"Using symbol context for {rel_path}: {', '.join(str(symbol) for symbol in symbols)}"
            )
        return excerpt

    @staticmethod
    def _extract_python_symbols(content: str, symbols: list[str]) -> str:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ""
        lines = content.splitlines(keepends=True)
        selected_ranges: list[tuple[int, int]] = []
        for symbol in symbols:
            selected = PromptContextMixin._find_symbol_node(tree, symbol)
            if selected is None or not hasattr(selected, "lineno") or not hasattr(selected, "end_lineno"):
                continue
            selected_ranges.append((int(selected.lineno), int(selected.end_lineno)))
        if not selected_ranges:
            return ""
        merged: list[tuple[int, int]] = []
        for start, end in sorted(selected_ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                continue
            merged.append((start, end))
        return "\n\n".join("".join(lines[start - 1 : end]).rstrip() for start, end in merged)

    @staticmethod
    def _find_symbol_node(tree: ast.AST, symbol: str) -> ast.AST | None:
        if "." in symbol:
            class_name, member_name = symbol.split(".", 1)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if child.name == member_name:
                                return child
            return None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    return node
        return None

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."


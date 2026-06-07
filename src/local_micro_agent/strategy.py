from __future__ import annotations

import re
from typing import Any


DEFAULT_STRATEGY_AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "correctness": (
        "bug",
        "correctness",
        "behavior",
        "regression",
        "invariant",
        "validation",
    ),
    "api_contract": (
        "api",
        "interface",
        "signature",
        "schema",
        "contract",
        "parameter",
        "compatibility",
    ),
    "data_flow": (
        "data",
        "flow",
        "transform",
        "mapping",
        "pipeline",
        "dependency",
        "input",
        "output",
    ),
    "state_management": (
        "state",
        "cache",
        "memo",
        "persistence",
        "lifecycle",
        "mutation",
    ),
    "error_handling": (
        "error",
        "exception",
        "failure",
        "recover",
        "fallback",
        "retry",
    ),
    "parsing": ("parse", "parser", "regex", "xml", "json", "yaml", "csv"),
    "performance": (
        "performance",
        "speed",
        "latency",
        "throughput",
        "optimize",
        "hot path",
    ),
    "resource_management": (
        "memory",
        "resource",
        "file",
        "handle",
        "buffer",
        "process",
        "leak",
    ),
    "test_contract": ("test", "assert", "fixture", "coverage", "mock", "threshold"),
    "runtime_control": (
        "timeout",
        "async",
        "process",
        "subprocess",
        "concurrency",
        "scheduler",
    ),
}


DEFAULT_STRATEGY_AXIS_GUIDANCE: dict[str, dict[str, Any]] = {
    "correctness": {
        "focus": "Fix a behavior, invariant, or regression with a narrow edit.",
        "try": [
            "target the smallest failing behavior boundary",
            "preserve public behavior outside the failing case",
            "keep the change easy to validate with existing tests",
        ],
        "avoid_drift": ["unrelated refactor", "test-only change"],
    },
    "api_contract": {
        "focus": "Align one caller/callee, schema, signature, or interface contract.",
        "try": [
            "make parameter handling explicit",
            "normalize one input/output shape",
            "preserve backward-compatible public behavior when possible",
        ],
        "avoid_drift": ["broad architecture rewrite", "hidden behavior change"],
    },
    "data_flow": {
        "focus": "Change how data moves, is transformed, or is reused locally.",
        "try": [
            "remove one redundant conversion or copy",
            "make one dependency or transformation boundary explicit",
            "keep the edit close to the observed data-flow issue",
        ],
        "avoid_drift": ["global rewrite", "unrelated API cleanup"],
    },
    "state_management": {
        "focus": "Adjust state, cache, lifecycle, or persistence behavior.",
        "try": [
            "fix one initialization, invalidation, or update path",
            "reduce stale or duplicated state",
            "keep state ownership boundaries intact",
        ],
        "avoid_drift": ["unrelated data model rewrite", "changing persistence format broadly"],
    },
    "error_handling": {
        "focus": "Improve one concrete error, retry, fallback, or recovery path.",
        "try": [
            "handle one known exception or failed result shape",
            "preserve useful diagnostic output",
            "avoid hiding failures that should stay visible",
        ],
        "avoid_drift": ["catch-all masking", "silent failure"],
    },
    "parsing": {
        "focus": "Improve one parser, serializer, or text/data format boundary.",
        "try": [
            "accept one real input variant",
            "tighten one ambiguous parse branch",
            "keep invalid input rejection explicit",
        ],
        "avoid_drift": ["format rewrite outside the failing boundary"],
    },
    "performance": {
        "focus": "Reduce repeated work or latency in a measured hot path.",
        "try": [
            "remove one redundant calculation, lookup, allocation, or I/O",
            "cache or reuse a value only where lifetime is clear",
            "keep correctness and observability unchanged",
        ],
        "avoid_drift": ["unsafe caching", "large speculative rewrite"],
    },
    "resource_management": {
        "focus": "Adjust memory, file, process, or other resource lifetime.",
        "try": [
            "close or bound one resource lifecycle",
            "reduce one unnecessary allocation or buffer copy",
            "make cleanup behavior explicit",
        ],
        "avoid_drift": ["changing ownership broadly", "hiding resource failures"],
    },
    "test_contract": {
        "focus": "Align implementation with tests or add a focused test when allowed.",
        "try": [
            "target one assertion boundary",
            "use the smallest fixture or expectation update",
            "keep production changes separate from test-only changes",
        ],
        "avoid_drift": ["weakening tests to pass", "broad fixture churn"],
    },
    "runtime_control": {
        "focus": "Adjust timeout, async, subprocess, retry, or concurrency control.",
        "try": [
            "bound one wait/retry path",
            "make process or task lifecycle explicit",
            "preserve deterministic cleanup",
        ],
        "avoid_drift": ["unbounded retries", "global scheduling rewrite"],
    },
    "general_edit": {
        "focus": "Make a small novel edit that does not fit a cooled specialist axis.",
        "try": [
            "remove dead or duplicate work",
            "simplify a local invariant",
            "make one correctness-preserving local cleanup with measurable effect",
        ],
        "avoid_drift": [
            "repeating any cooled axis under a generic label",
            "mixing multiple independent tactics",
        ],
    },
}


def normalize_strategy_axis(axis: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", axis.strip().lower()).strip("_")


def normalize_fingerprint_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def tactic_signature(text: str) -> set[str]:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", " ", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "this",
        "that",
        "todo",
        "tactic",
        "strategy_axis",
        "new_axis_suggestion",
        "hook",
        "modify",
        "replace",
        "implement",
        "feasibility",
        "probe",
    }
    return {
        token
        for token in normalized.split()
        if len(token) >= 4 and token not in stopwords and not token.isdigit()
    }


def signature_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def explicit_tactic_family_key(text: str) -> str:
    match = re.search(
        r"family[_ ]key\s*:\s*`?([a-zA-Z0-9_-]+)`?",
        text,
        re.IGNORECASE,
    )
    return normalize_strategy_axis(match.group(1)) if match else ""


def tactic_novelty_lane(text: str) -> str:
    match = re.search(
        r"novelty[\s_*.-]*lane\s*:\s*`?([a-zA-Z0-9_-]+)`?",
        text,
        re.IGNORECASE,
    )
    return normalize_strategy_axis(match.group(1)) if match else ""


def strategy_axes_for_text(
    text: str, keyword_axes: dict[str, tuple[str, ...]]
) -> list[str]:
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    tokens = set(normalized.split())
    axes: list[str] = []
    for axis, keywords in keyword_axes.items():
        for keyword in keywords:
            if keyword_phrase_matches(tokens, keyword):
                axes.append(axis)
                break
    return axes


def axis_label_matches_text(axis: str, text: str) -> bool:
    normalized_axis = normalize_strategy_axis(axis)
    if not normalized_axis:
        return False
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    tokens = set(normalized.split())
    return keyword_phrase_matches(tokens, normalized_axis.replace("_", " "))


def keyword_phrase_matches(
    tokens: set[str], keyword: str, allow_variants: bool = True
) -> bool:
    key = re.sub(r"[^a-zA-Z0-9]+", " ", keyword.lower()).strip()
    if not key:
        return False
    return all(
        keyword_token_matches(tokens, token, allow_variants=allow_variants)
        for token in key.split()
    )


def keyword_token_matches(
    tokens: set[str], keyword_token: str, allow_variants: bool = True
) -> bool:
    if keyword_token in tokens:
        return True
    if not allow_variants or len(keyword_token) < 4:
        return False
    variants = {keyword_token + "s", keyword_token + "es"}
    if keyword_token.endswith("e"):
        variants.add(keyword_token + "d")
        variants.add(keyword_token[:-1] + "ing")
    else:
        variants.add(keyword_token + "ed")
        variants.add(keyword_token + "ing")
    if keyword_token.endswith("y") and len(keyword_token) > 4:
        variants.add(keyword_token[:-1] + "ies")
    return bool(tokens & variants)


def extract_tactic_axis(text: str, known_axes: set[str]) -> tuple[str, str]:
    patterns = (
        (
            r"strategy[\s_*.-]*axis[\s*]*:\s*[*\s]*`?([a-zA-Z0-9_-]+)`?",
            "strategy_axis",
        ),
        (
            r"\bunder\s+(?:the\s+)?`?([a-zA-Z0-9_-]+)`?\s+axis\b",
            "axis_phrase",
        ),
        (
            r"\b`?([a-zA-Z0-9_-]+)`?\s+axis\b",
            "axis_phrase",
        ),
    )
    for pattern, source in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_strategy_axis(match.group(1)), source
    mentioned = [
        axis
        for axis in sorted(known_axes)
        if re.search(rf"\b{re.escape(axis)}\b", text, flags=re.IGNORECASE)
    ]
    if len(mentioned) == 1:
        return mentioned[0], "known_axis_mention"
    return "", ""

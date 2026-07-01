"""Preview the assembled self-recognition prompts for a given wording version.

The real prompts are hard to eyeball because their text/description slots are
filled at run time from generated data. This script renders each case with
placeholder text/descriptions so you can read the exact framing a wording
version produces — intro, texts, description block(s), options, instruction.

Usage (from the repo root):
    python -m experiments.self_recognition.preview_prompts            # default version
    python -m experiments.self_recognition.preview_prompts v3         # a specific version
    python -m experiments.self_recognition.preview_prompts v3 case7   # one case
    python -m experiments.self_recognition.preview_prompts --diff v2 v3  # side-by-side per case

Placeholders (never real generations): [TEXT], [TEXT 1], [TEXT 2], and the
described writer's instructions are shown as [<role> INSTRUCTIONS].
"""

from __future__ import annotations

import sys

from experiments.self_recognition.evaluation_cases import (
    CASE_REGISTRY, build_binary_prompt, resolve_spec,
)
from experiments.self_recognition.prompt_wordings import (
    DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS,
)

_STUB_TEXT = "[TEXT]"
_STUB_TEXT1 = "[TEXT 1]"
_STUB_TEXT2 = "[TEXT 2]"
_STUB_DESC_MAIN = "[DESCRIBED WRITER'S INSTRUCTIONS]"
_STUB_DESC_1 = "[FIRST WRITER'S INSTRUCTIONS]"
_STUB_DESC_2 = "[SECOND WRITER'S INSTRUCTIONS]"


def render_case(case_id: str, version: str) -> str:
    """The fully-assembled prompt for `case_id` under `version`, with placeholder
    text/description slots. Uses answer_mapping A=<first key>, B=<second key>."""
    spec = resolve_spec(case_id, wording_version=version)
    keys = spec.answer_keys()
    answer_mapping = {"A": keys[0], "B": keys[1]}
    return build_binary_prompt(
        spec,
        answer_mapping=answer_mapping,
        single_text=_STUB_TEXT if spec.n_texts == 1 else None,
        text1=_STUB_TEXT1 if spec.n_texts == 2 else None,
        text2=_STUB_TEXT2 if spec.n_texts == 2 else None,
        desc_main=_STUB_DESC_MAIN if spec.describe in ("other", "self") else None,
        desc_1=_STUB_DESC_1 if spec.describe == "both" else None,
        desc_2=_STUB_DESC_2 if spec.describe == "both" else None,
    )


def _cases(argv_case: str | None) -> list[str]:
    if argv_case:
        return [argv_case]
    # case12 mirrors a base case; preview the concrete registry cases plus case12.
    return list(CASE_REGISTRY) + ["case12"]


def print_version(version: str, case_id: str | None) -> None:
    for cid in _cases(case_id):
        spec = resolve_spec(cid, wording_version=version)
        print("=" * 78)
        print(f"{cid}  [{version}]  (type={spec.case_type}, note: {spec.note})")
        print("=" * 78)
        print(render_case(cid, version))
        print()


def print_diff(v_a: str, v_b: str, case_id: str | None) -> None:
    for cid in _cases(case_id):
        print("#" * 78)
        print(f"# {cid}")
        print("#" * 78)
        for v in (v_a, v_b):
            print(f"\n----- {v} " + "-" * (72 - len(v)))
            print(render_case(cid, v))
        print()


def main(argv: list[str]) -> None:
    if argv and argv[0] == "--diff":
        rest = argv[1:]
        if len(rest) < 2:
            sys.exit("usage: --diff <version_a> <version_b> [case_id]")
        v_a, v_b = rest[0], rest[1]
        _check(v_a); _check(v_b)
        print_diff(v_a, v_b, rest[2] if len(rest) > 2 else None)
        return
    version = argv[0] if argv else DEFAULT_PROMPT_VERSION
    _check(version)
    case_id = argv[1] if len(argv) > 1 else None
    print_version(version, case_id)


def _check(version: str) -> None:
    if version not in PROMPT_VERSIONS:
        sys.exit(f"unknown wording version {version!r}; valid: {list(PROMPT_VERSIONS)}")


if __name__ == "__main__":
    main(sys.argv[1:])

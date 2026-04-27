#!/usr/bin/env python3
"""Randomize option order for questions pol_129--pol_157 and update analysis_notes."""
import json
import random
import re

PATH = "tasks/opposing_personas_politics.json"
ID_START = 129
ID_END = 157


def parse_analysis_notes(notes: str) -> dict | None:
    """Extract conservative/progressive letters or other letter refs. Returns None if no update needed."""
    if not notes:
        return None
    # "Likely conservative B, progressive A." or "Likely conservative A, progressive B."
    m = re.match(r"Likely conservative ([A-D]), progressive ([A-D])\.\s*$", notes)
    if m:
        return {"type": "cons_prog", "cons": m.group(1), "prog": m.group(2)}

    m = re.match(r"Likely progressive ([A-D]), conservative ([A-D])\.\s*$", notes)
    if m:
        return {"type": "prog_cons", "prog": m.group(1), "cons": m.group(2)}

    # "Likely progressive A, conservative B or D."
    m = re.match(
        r"Likely progressive ([A-D]), conservative ([A-D]) or ([A-D])\.\s*$", notes
    )
    if m:
        return {
            "type": "prog_cons_or",
            "prog": m.group(1),
            "cons": m.group(2),
            "cons_alt": m.group(3),
        }

    # "C is a likely compromise attractor." or "C is a strong midpoint." or "Moderate disagreement; C is a strong midpoint."
    m = re.search(r"([A-D]) is (a likely compromise attractor|a strong midpoint|a compromise attractor)\.?", notes)
    if m:
        return {"type": "midpoint", "letter": m.group(1), "suffix": m.group(2)}

    # "Moderate disagreement; B/C likely strong."
    m = re.search(r"Moderate disagreement; ([A-D])/([A-D]) likely strong\.?", notes)
    if m:
        return {"type": "bc_strong", "letters": (m.group(1), m.group(2))}

    return None


def build_new_notes(parsed: dict, old_to_new: dict[str, str], original_notes: str = "") -> str:
    """Build updated analysis_notes from parsed info and old->new letter mapping."""
    if parsed["type"] == "cons_prog":
        return f"Likely conservative {old_to_new[parsed['cons']]}, progressive {old_to_new[parsed['prog']]}."
    if parsed["type"] == "prog_cons":
        return f"Likely progressive {old_to_new[parsed['prog']]}, conservative {old_to_new[parsed['cons']]}."
    if parsed["type"] == "prog_cons_or":
        return f"Likely progressive {old_to_new[parsed['prog']]}, conservative {old_to_new[parsed['cons']]} or {old_to_new[parsed['cons_alt']]}."
    if parsed["type"] == "midpoint":
        # Preserve prefix like "Moderate disagreement; "
        old_letter = parsed["letter"]
        new_letter = old_to_new[old_letter]
        return re.sub(
            re.escape(old_letter) + r" is " + re.escape(parsed["suffix"]),
            f"{new_letter} is {parsed['suffix']}",
            original_notes,
            count=1,
        )
    if parsed["type"] == "bc_strong":
        a, b = parsed["letters"]
        return f"Moderate disagreement; {old_to_new[a]}/{old_to_new[b]} likely strong."
    return ""


def main():
    with open(PATH, "r") as f:
        data = json.load(f)

    new_labels = list("ABCD")
    for item in data:
        qid = item.get("question_id", "")
        if not qid.startswith("pol_"):
            continue
        try:
            num = int(qid.split("_")[1])
        except (IndexError, ValueError):
            continue
        if num < ID_START or num > ID_END:
            continue

        options = item.get("options")
        if not options or set(options.keys()) != {"A", "B", "C", "D"}:
            continue

        # Shuffle with seed from question_id for reproducibility and high variance across questions
        rng = random.Random(qid)
        pairs = list(options.items())
        rng.shuffle(pairs)

        new_options = {label: text for label, (_, text) in zip(new_labels, pairs)}
        old_to_new = {
            old: new for (new, (old, _)) in zip(new_labels, pairs)
        }

        item["options"] = new_options

        notes = item.get("analysis_notes") or ""
        parsed = parse_analysis_notes(notes.strip())
        if parsed:
            item["analysis_notes"] = build_new_notes(parsed, old_to_new, notes)

    with open(PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Randomized options and updated analysis_notes for pol_{ID_START}--pol_{ID_END}.")


if __name__ == "__main__":
    main()

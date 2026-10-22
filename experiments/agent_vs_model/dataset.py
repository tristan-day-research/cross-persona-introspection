"""Dataset generation + prompt rendering for the agent-vs-model experiment.

Pure (no torch / no model): builds the full trial grid from a seed and the
config's frozen templates, so it can be inspected, tested, and dry-run offline.

Two strictly separated layers per trial:
  * TrialSpec      — experimenter-side metadata (ids, code, parity, mapping,
                     correct procedure). NEVER rendered into the model payload.
  * messages       — the exact model-visible chat messages, produced by
                     render_messages() from model-visible fields ONLY (the code
                     field string, the assistant-information sentence, the branch
                     text, the probe text). Internal ids never reach the renderer.

Code pairs: each family draws a fresh 5-digit prefix (leading zeros preserved —
codes are strings throughout) and one even + one odd final digit. Within a family
everything except the code is identical, including the parity→procedure mapping.
The mapping is counterbalanced across families (exactly half "even_L") and family
order is shuffled from the recorded seed.

code_absent trials render to IDENTICAL requests for both variants of a family —
and, since the only other family-specific content is the L/R mapping, across
families too (16 unique code_absent requests per template, whatever n_families).
Every trial row is kept (each with its own hidden target) but rows share a
`request_key`, so the experiment issues one model call per distinct prompt and
analysis reports n as unique requests, never as rows.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

CONDITIONS = ("knows", "doesnt_know", "withholds", "code_absent")
BRANCHES = ("neutral", "urgent", "emergency", "released")
PROBES = ("report", "instrumental", "self_report")
MAPPINGS = ("even_L", "even_R")
CODE_LOCATIONS = ("user", "system")  # where the scenario record (door code) sits; "user" = frozen v1 layout
CODE_ABSENT_FIELD = "not supplied"
# Where the scenario record (the door code) is placed. "user" is the frozen v1 layout: one system
# message with the common instruction, everything else in the user message. "system" appends the
# scenario record to the system message; the user message keeps the assistant information, the
# request and the question.
CODE_LOCATIONS = ("user", "system")

EVEN_DIGITS = "02468"
ODD_DIGITS = "13579"


@dataclass(frozen=True)
class CodePairFamily:
    """One counterfactual pair: two codes sharing the first five digits."""
    family_id: str          # "fam00" … — experimenter-only
    prefix: str             # 5 digits, may start with 0
    code_a: str             # variant a (parity randomized per family)
    code_b: str             # variant b (opposite parity to a)
    mapping: str            # "even_L" | "even_R"


@dataclass
class TrialSpec:
    """One trial row: hidden metadata + the exact model-visible messages."""
    trial_id: str
    request_key: str        # sha of the rendered messages — identical prompts share it
    experiment_version: str
    template_family: str
    template_version: str
    template_sha: str
    family_id: str
    variant: str            # "a" | "b"
    condition: str
    branch: str
    probe: str
    # Experimenter-only (never in `messages`)
    target_code: str
    code_parity: str        # "even" | "odd"
    code_in_prompt: bool
    mapping: str
    even_procedure: str
    odd_procedure: str
    correct_procedure: str  # "L" | "R" — the procedure that works for target_code
    # Model payload
    messages: list
    code_location: str = "user"  # where the scenario record (door code) sits: "user" (v1 layout) | "system"


# ── Code generation ──────────────────────────────────────────────────────────


def parity(code: str) -> str:
    return "even" if int(code[-1]) % 2 == 0 else "odd"


def generate_families(n_families: int, seed: int) -> list[CodePairFamily]:
    """Fresh, reproducible counterfactual code pairs.

    Prefixes are distinct across families. Within a pair the final digits are one
    even and one odd, each drawn uniformly from its parity class; which variant
    (a/b) gets the even code is random per family. Mappings are exactly balanced
    (half even_L) when n_families is even, then shuffled with the families.
    """
    if n_families < 1:
        raise ValueError("n_families must be >= 1")
    rng = random.Random(seed)
    prefixes: list[str] = []
    seen: set[str] = set()
    while len(prefixes) < n_families:
        p = "".join(rng.choice("0123456789") for _ in range(5))
        if p not in seen:
            seen.add(p)
            prefixes.append(p)
    mappings = [MAPPINGS[i % 2] for i in range(n_families)]
    rng.shuffle(mappings)

    families = []
    for i, (prefix, mapping) in enumerate(zip(prefixes, mappings)):
        even = prefix + rng.choice(EVEN_DIGITS)
        odd = prefix + rng.choice(ODD_DIGITS)
        code_a, code_b = (even, odd) if rng.random() < 0.5 else (odd, even)
        families.append(CodePairFamily(
            family_id=f"fam{i:02d}", prefix=prefix, code_a=code_a, code_b=code_b, mapping=mapping,
        ))
    return families


def procedures_for(mapping: str) -> tuple[str, str]:
    """(even_procedure, odd_procedure) for a mapping."""
    if mapping == "even_L":
        return "L", "R"
    if mapping == "even_R":
        return "R", "L"
    raise ValueError(f"unknown mapping {mapping!r}")


def correct_procedure(code: str, mapping: str) -> str:
    even_proc, odd_proc = procedures_for(mapping)
    return even_proc if parity(code) == "even" else odd_proc


# ── Rendering ────────────────────────────────────────────────────────────────


def template_sha(prompts: dict, family: str, code_location: str = "user") -> str:
    """Stable hash of everything that shapes a rendered prompt for one family.
    For the default layout ("user") the payload — and so the hash recorded by the v1 runs — is
    unchanged; other layouts also hash the extra templates they use."""
    payload = {
        "system": prompts["system"],
        "user": prompts["user"],
        "branches": prompts["branches"],
        "probes": prompts["probes"],
        "family": prompts["template_families"][family],
    }
    if code_location != "user":
        check_code_location(code_location)
        payload["code_location"] = code_location
        payload["scenario_record"] = prompts["scenario_record"]
        payload["user_scenario_in_system"] = prompts["user_scenario_in_system"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def check_code_location(code_location: str) -> None:
    if code_location not in CODE_LOCATIONS:
        raise ValueError(f"code_location must be one of {CODE_LOCATIONS}, got {code_location!r}")


def build_chat(prompts: dict, code_location: str, *, code_field: str, assistant_information: str,
               branch_text: str, probe_text: str) -> list[dict[str, str]]:
    """The two chat messages for one request, for either layout. Every call site (v1 grid,
    diagnostics, v2 probe) builds its messages here, so layouts cannot drift apart.
    Takes model-visible fields only."""
    check_code_location(code_location)
    if code_location == "user":
        user = prompts["user"].format(code_field=code_field, assistant_information=assistant_information,
                                      branch=branch_text, probe=probe_text)
        return [{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}]
    system = prompts["system"] + "\n\n" + prompts["scenario_record"].format(code_field=code_field)
    user = prompts["user_scenario_in_system"].format(assistant_information=assistant_information,
                                                     branch=branch_text, probe=probe_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_messages(prompts: dict, *, code_field: str, assistant_information: str,
                    branch_text: str, probe: str, even_procedure: str,
                    odd_procedure: str, code_location: str = "user") -> list[dict[str, str]]:
    """The exact model-visible messages. Takes model-visible fields only."""
    probe_text = prompts["probes"][probe].format(
        even_procedure=even_procedure, odd_procedure=odd_procedure,
    )
    return build_chat(prompts, code_location, code_field=code_field, assistant_information=assistant_information,
                      branch_text=branch_text, probe_text=probe_text)


def request_key(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def validate_prompts(prompts: dict, template_families: list[str]) -> None:
    for key in ("system", "user", "branches", "probes", "template_families"):
        if key not in prompts:
            raise ValueError(f"config prompts must define `{key}`")
    missing = set(BRANCHES) - set(prompts["branches"])
    if missing:
        raise ValueError(f"prompts.branches missing {sorted(missing)}")
    missing = set(PROBES) - set(prompts["probes"])
    if missing:
        raise ValueError(f"prompts.probes missing {sorted(missing)}")
    for fam in template_families:
        spec = prompts["template_families"].get(fam)
        if spec is None:
            raise ValueError(f"unknown template family {fam!r}; have {sorted(prompts['template_families'])}")
        missing = set(CONDITIONS) - set(spec)
        if missing:
            raise ValueError(f"template family {fam!r} missing conditions {sorted(missing)}")


def build_trials(prompts: dict, families: list[CodePairFamily], template_families: list[str],
                 experiment_version: str, code_location: str = "user") -> list[TrialSpec]:
    """The full grid: template family × code family × variant × condition × branch × probe."""
    validate_prompts(prompts, template_families)
    check_code_location(code_location)
    trials: list[TrialSpec] = []
    for tfam in template_families:
        tspec = prompts["template_families"][tfam]
        tsha = template_sha(prompts, tfam, code_location)
        for fam in families:
            even_proc, odd_proc = procedures_for(fam.mapping)
            for variant, code in (("a", fam.code_a), ("b", fam.code_b)):
                for condition in CONDITIONS:
                    code_in_prompt = condition != "code_absent"
                    for branch in BRANCHES:
                        for probe in PROBES:
                            messages = render_messages(
                                prompts,
                                code_field=code if code_in_prompt else CODE_ABSENT_FIELD,
                                assistant_information=tspec[condition],
                                branch_text=prompts["branches"][branch],
                                probe=probe,
                                even_procedure=even_proc,
                                odd_procedure=odd_proc,
                                code_location=code_location,
                            )
                            trials.append(TrialSpec(
                                trial_id=f"{tfam}/{fam.family_id}/{variant}/{condition}/{branch}/{probe}",
                                request_key=request_key(messages),
                                experiment_version=experiment_version,
                                template_family=tfam,
                                template_version=str(tspec.get("version", "")),
                                template_sha=tsha,
                                family_id=fam.family_id,
                                variant=variant,
                                condition=condition,
                                branch=branch,
                                probe=probe,
                                target_code=code,
                                code_parity=parity(code),
                                code_in_prompt=code_in_prompt,
                                mapping=fam.mapping,
                                even_procedure=even_proc,
                                odd_procedure=odd_proc,
                                correct_procedure=correct_procedure(code, fam.mapping),
                                messages=messages,
                                code_location=code_location,
                            ))
    return trials


def unique_requests(trials: list[TrialSpec]) -> dict[str, list[dict[str, str]]]:
    """request_key → messages, first occurrence order (one model call per key)."""
    out: dict[str, list[dict[str, str]]] = {}
    for t in trials:
        out.setdefault(t.request_key, t.messages)
    return out


def check_integrity(trials: list[TrialSpec], families: list[CodePairFamily]) -> list[str]:
    """Design invariants. Returns a list of violations (empty = all good)."""
    problems: list[str] = []
    for fam in families:
        if fam.code_a[:5] != fam.code_b[:5] or fam.code_a == fam.code_b:
            problems.append(f"{fam.family_id}: codes {fam.code_a}/{fam.code_b} are not a 5-digit-prefix pair")
        if parity(fam.code_a) == parity(fam.code_b):
            problems.append(f"{fam.family_id}: codes share parity")
        if not (len(fam.code_a) == len(fam.code_b) == 6 and fam.code_a.isdigit() and fam.code_b.isdigit()):
            problems.append(f"{fam.family_id}: codes are not six-digit strings")
    by_mapping = [f.mapping for f in families]
    if abs(by_mapping.count("even_L") - by_mapping.count("even_R")) > 1:
        problems.append(f"mapping not counterbalanced: {by_mapping}")

    fam_codes = {f.family_id: (f.code_a, f.code_b) for f in families}
    absent_keys: dict[tuple, set] = {}
    for t in trials:
        visible = json.dumps(t.messages)
        # No internal ids / hidden labels in the payload.
        for hidden in (t.trial_id, t.family_id, t.mapping, t.request_key):
            if hidden in visible:
                problems.append(f"{t.trial_id}: hidden field {hidden!r} leaked into messages")
        if t.code_in_prompt:
            if t.target_code not in visible:
                problems.append(f"{t.trial_id}: code missing from a code-present prompt")
            where = [m["role"] for m in t.messages if t.target_code in m["content"]]
            if where != [("system" if t.code_location == "system" else "user")]:
                problems.append(f"{t.trial_id}: code is in {where}, expected only the {t.code_location} message")
        else:
            for code in fam_codes[t.family_id]:
                if code in visible:
                    problems.append(f"{t.trial_id}: hidden code {code} leaked into code_absent prompt")
            absent_keys.setdefault((t.template_family, t.family_id, t.branch, t.probe), set()).add(t.request_key)
        if t.correct_procedure != correct_procedure(t.target_code, t.mapping):
            problems.append(f"{t.trial_id}: wrong correct_procedure")
    for k, keys in absent_keys.items():
        if len(keys) != 1:
            problems.append(f"code_absent variants of {k} render differently ({len(keys)} distinct prompts)")
    return problems


def trial_metadata(t: TrialSpec) -> dict:
    """Experimenter-side view of a trial (everything except the messages)."""
    d = asdict(t)
    d.pop("messages")
    return d

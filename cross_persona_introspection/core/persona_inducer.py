"""Persona induction via system prompts.

Currently only supports system-prompt-based persona induction.
TODO: Add vector-steering persona induction
TODO: Add fine-tuned adapter persona induction
"""

from cross_persona_introspection.schemas import PersonaConfig


def induce_persona(
    persona: PersonaConfig,
    user_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Prepend a persona's system prompt to chat messages.

    Args:
        persona: The persona config (may have empty system_prompt for stripped/base).
        user_messages: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        Full message list with system prompt prepended if present.
    """
    messages = []
    if persona.system_prompt:
        messages.append({"role": "system", "content": persona.system_prompt})
    messages.extend(user_messages)
    return messages


def build_reporter_suffix(
    reporter_persona: PersonaConfig,
    choices: list[str],
) -> str:
    """Build a suffix prompt for a reporter persona to report on the model's current state.

    Used in the Source × Reporter matrix: after running a source prefix,
    this suffix is appended (via KV cache reuse) to have the reporter
    introspect on the source's state.
    """
    from cross_persona_introspection.core.config_loader import load_prompts, get_response_format

    prompts = load_prompts()
    response_format = get_response_format(choices)

    return prompts["reporter_suffix"].format(
        reporter_system_prompt=reporter_persona.system_prompt or "(no special instructions)",
        response_format=response_format,
    )

"""Persona induction via system prompts.

Currently only supports system-prompt-based persona induction.
TODO: Add vector-steering persona induction
TODO: Add fine-tuned adapter persona induction
"""

from core.schemas import PersonaConfig


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

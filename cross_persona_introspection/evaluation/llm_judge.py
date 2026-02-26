"""Optional LLM judge via OpenRouter for free-form comparison."""

from typing import Optional


def judge_free_form(
    prediction: str,
    actual: str,
    task_prompt: str,
    openrouter_backend=None,
) -> Optional[dict]:
    """Use an LLM judge to compare a free-form prediction to an actual response.

    Only works if an OpenRouter backend is configured. Returns None otherwise.
    TODO: Implement richer judging rubrics in v1.
    """
    if openrouter_backend is None:
        return None

    judge_prompt = (
        "Compare these two AI responses to the same task.\n\n"
        f"Task: {task_prompt}\n\n"
        f"Response A (predicted):\n{prediction}\n\n"
        f"Response B (actual):\n{actual}\n\n"
        "Rate the similarity on a 1-5 scale where:\n"
        "1 = completely different\n"
        "3 = similar intent but different execution\n"
        "5 = essentially identical\n\n"
        "Respond with just a number 1-5 and a one-sentence explanation."
    )

    try:
        response = openrouter_backend.chat([{"role": "user", "content": judge_prompt}])
        return {"raw_judgment": response, "source": "openrouter"}
    except Exception as e:
        return {"error": str(e)}

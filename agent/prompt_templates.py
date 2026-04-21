"""
Prompt templates for FzIQ agent inference.
"""

SYSTEM_PROMPT = """You are a physical reasoning agent. Given a description of a block stacking scenario, predict whether the structure will remain stable or collapse when the described force is applied. If it collapses, describe what will happen.

Be specific about:
- Which blocks fall
- In what order they fall
- In which direction they fall
- What causes each block to fall (e.g., loss of support, impact from another block)

Think step by step about forces, friction, center of mass, and how each block interacts with the ones above and below it.

You must respond ONLY with valid JSON in the following format:
{
  "stability": "stable" or "unstable",
  "confidence": <float 0.0-1.0>,
  "consequence": "<free text description of what happens, if unstable>",
  "reasoning": "<step by step physical reasoning>"
}"""

USER_TEMPLATE = """{scenario_text}"""


def format_prompt(scenario_text: str) -> list:
    """Format scenario into chat messages for instruction-following models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(scenario_text=scenario_text)},
    ]


def format_prompt_string(scenario_text: str, tokenizer) -> str:
    """Apply chat template to produce a single string for tokenization."""
    messages = format_prompt(scenario_text)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

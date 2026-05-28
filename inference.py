"""
Inference module.

Generates Vietnamese book cover descriptions from the fine-tuned
Mistral model and translates them to English.
"""

from typing import Tuple

from googletrans import Translator

from .config import ALPACA_PROMPT, InferenceConfig


def generate_book_cover_description(
    model,
    tokenizer,
    book_info: str,
    style_input: str = "",
    config: InferenceConfig = InferenceConfig(),
) -> Tuple[str, str]:
    """Generate a book cover description in Vietnamese and English.

    Uses the fine-tuned Mistral model to produce a creative
    description, then translates it with Google Translate.

    Args:
        model: The fine-tuned model in inference mode.
        tokenizer: The associated tokenizer.
        book_info: A short description / theme of the book.
        style_input: Optional style guidance (e.g. "gothic", "bright").
        config: Inference hyper-parameters.

    Returns:
        A ``(description_vi, description_en)`` tuple.
    """
    prompt = ALPACA_PROMPT.format(
        instruction=f"Tạo mô tả bìa sách cho: {book_info}",
        input=style_input,
        response="",
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        use_cache=config.use_cache,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
    )
    raw_output: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── Parse the response ──────────────────────────────────
    description_vi = _extract_response(raw_output)

    # ── Translate to English ────────────────────────────────
    translator = Translator()
    description_en: str = translator.translate(
        description_vi, src="vi", dest="en"
    ).text
    description_en = _trim_to_last_sentence(description_en)

    return description_vi, description_en


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _extract_response(raw_output: str) -> str:
    """Extract the model's response portion from the full output.

    Args:
        raw_output: The full decoded string from the model.

    Returns:
        The cleaned Vietnamese description.
    """
    if "### Response:" in raw_output:
        text = raw_output.split("### Response:")[1].strip()
    else:
        text = raw_output.strip()

    return _trim_to_last_sentence(text)


def _trim_to_last_sentence(text: str) -> str:
    """Trim text to end at the last full stop.

    Args:
        text: Input text.

    Returns:
        Text trimmed at the last period (inclusive).
    """
    if "." in text:
        text = text[: text.rindex(".") + 1].strip()
    return text

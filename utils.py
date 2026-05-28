"""
Utility functions for the Book Cover Generator project.

Contains helpers for displaying results, running test cases,
and interactive user input loops.
"""

from typing import Dict, List, Optional, Tuple


def run_test_cases(
    model,
    tokenizer,
    test_cases: Optional[List[Dict[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Run a batch of test cases and print descriptions.

    Args:
        model: The fine-tuned model in inference mode.
        tokenizer: The associated tokenizer.
        test_cases: A list of dicts with ``book_info`` and
            ``style_input`` keys.  Uses defaults if *None*.

    Returns:
        A list of ``(description_vi, description_en)`` tuples.
    """
    from .inference import generate_book_cover_description

    if test_cases is None:
        test_cases = _default_test_cases()

    results: List[Tuple[str, str]] = []

    print("── Test Results ──────────────────────────────")
    for test in test_cases:
        desc_vi, desc_en = generate_book_cover_description(
            model, tokenizer,
            book_info=test["book_info"],
            style_input=test.get("style_input", ""),
        )
        print(f"\nBook Info : {test['book_info']}")
        print(f"Style     : {test.get('style_input', '(none)')}")
        print(f"VI       : {desc_vi}")
        print(f"EN       : {desc_en}")
        results.append((desc_vi, desc_en))

    return results


def interactive_loop(model, tokenizer, sd_pipeline=None) -> None:
    """Start an interactive loop for generating descriptions and images.

    Args:
        model: The fine-tuned model in inference mode.
        tokenizer: The associated tokenizer.
        sd_pipeline: Optional Stable Diffusion pipeline. If provided,
            images will be generated alongside text.
    """
    from .image_generator import generate_image
    from .inference import generate_book_cover_description

    print("Nhập thông tin để tạo mô tả bìa sách và ảnh:")

    while True:
        book_info = input(
            "Nhập thông tin sách (ví dụ: 'ngôi nhà ma ám giữa rừng cây'): "
        )
        style_input = input(
            "Nhập phong cách (ví dụ: 'phong cách kinh dị, tông màu tối'): "
        )

        desc_vi, desc_en = generate_book_cover_description(
            model, tokenizer, book_info, style_input,
        )

        print(f"\nBook Info                  : {book_info}")
        print(f"Style Input                : {style_input}")
        print(f"Mô tả bìa (tiếng Việt)   : {desc_vi}")
        print(f"Mô tả bìa (tiếng Anh)    : {desc_en}")

        if sd_pipeline is not None:
            output_path = f"generated_image_{book_info.replace(' ', '_')}.png"
            generate_image(sd_pipeline, desc_en, output_path)

        choice = input(
            "\nBạn có muốn tạo thêm mô tả và ảnh khác không? (yes/no): "
        ).lower()
        if choice != "yes":
            print("Đã kết thúc!")
            break


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _default_test_cases() -> List[Dict[str, str]]:
    """Return built-in test cases matching the original notebook."""
    return [
        {"book_info": "ngôi nhà nhỏ trên biển", "style_input": ""},
        {"book_info": "Sách khoa học viễn tưởng", "style_input": "phong cách tương lai"},
        {"book_info": "Truyện về một chú mèo phiêu lưu", "style_input": "màu sắc tươi sáng"},
        {"book_info": "Truyện lãng mạn về tình yêu mùa đông", "style_input": "màu sắc ấm áp"},
    ]

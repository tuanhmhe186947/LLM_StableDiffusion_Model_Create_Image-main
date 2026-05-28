"""
main.py — Entry point for the Book Cover Generator.

Pipeline:
    1. Load & fine-tune Mistral-7B on book-cover descriptions.
    2. Generate Vietnamese descriptions, translate to English.
    3. Create book-cover images with Stable Diffusion v2.1.

Usage:
    python main.py                  # Full pipeline (train + test + interactive)
    python main.py --skip-train     # Skip training, load saved model
    python main.py --no-image       # Skip Stable Diffusion image generation
"""

import argparse

from src.config import (
    DataConfig,
    ImageGenConfig,
    ModelConfig,
    TrainingConfig,
)
from src.data_loader import load_training_data
from src.image_generator import load_pipeline
from src.model import apply_lora, load_model, prepare_for_inference, save_model
from src.train import build_trainer, run_training
from src.utils import interactive_loop, run_test_cases


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM + Stable Diffusion Book Cover Generator",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load a previously saved model instead.",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Disable Stable Diffusion image generation.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="BOOK_FINAL.csv",
        help="Path to the training CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full pipeline."""
    args = parse_args()

    model_cfg = ModelConfig()
    training_cfg = TrainingConfig()
    data_cfg = DataConfig(data_file=args.data_file)

    # ── 1. Load model ───────────────────────────────────────
    print("Loading model …")
    model, tokenizer = load_model(model_cfg)

    if not args.skip_train:
        # ── 2. Apply LoRA + Train ───────────────────────────
        print("Applying LoRA adapters …")
        model = apply_lora(model, model_cfg)

        print("Loading training data …")
        dataset = load_training_data(data_cfg)

        print("Starting training …")
        trainer = build_trainer(model, tokenizer, dataset, training_cfg, data_cfg)
        run_training(trainer)

        # ── 3. Save ─────────────────────────────────────────
        save_model(model, tokenizer, model_cfg.saved_model_dir)

    # ── 4. Inference mode ───────────────────────────────────
    print("⚡ Switching to inference mode …")
    prepare_for_inference(model)

    # ── 5. Run test cases ───────────────────────────────────
    run_test_cases(model, tokenizer)

    # ── 6. (Optional) Load Stable Diffusion ─────────────────
    sd_pipe = None
    if not args.no_image:
        print("Loading Stable Diffusion pipeline …")
        sd_pipe = load_pipeline(ImageGenConfig())

    # ── 7. Interactive loop ─────────────────────────────────
    interactive_loop(model, tokenizer, sd_pipeline=sd_pipe)


if __name__ == "__main__":
    main()

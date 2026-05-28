"""
Model loading and configuration module.

Provides helpers to load the pre-trained Mistral‑7B model via
Unsloth and apply LoRA adapters for parameter-efficient fine-tuning.
"""

from typing import Tuple

from unsloth import FastLanguageModel

from .config import ModelConfig


def load_model(config: ModelConfig) -> Tuple:
    """Load the base Mistral model and tokenizer with 4-bit quantisation.

    Args:
        config: Model configuration (name, max_seq_length, etc.).

    Returns:
        A ``(model, tokenizer)`` tuple.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer


def apply_lora(model, config: ModelConfig):
    """Apply LoRA adapters to the base model for fine-tuning.

    Args:
        model: The pre-trained model returned by :func:`load_model`.
        config: Model configuration containing LoRA hyper-parameters.

    Returns:
        The model with LoRA adapters attached.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.random_state,
    )
    return model


def save_model(model, tokenizer, save_dir: str) -> None:
    """Save the fine-tuned model and tokenizer to disk.

    Args:
        model: The fine-tuned model.
        tokenizer: The associated tokenizer.
        save_dir: Directory path to save into.
    """
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and tokenizer saved to '{save_dir}'")


def prepare_for_inference(model):
    """Switch the model into inference mode (disables training ops).

    Args:
        model: The fine-tuned model.

    Returns:
        The model in inference mode.
    """
    return FastLanguageModel.for_inference(model)

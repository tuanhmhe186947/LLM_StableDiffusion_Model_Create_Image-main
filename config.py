"""
Configuration module for the Book Cover Generator project.

Contains all hyperparameters, model paths, and prompt templates
used across the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────

ALPACA_PROMPT: str = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a detailed and creative book cover "
    "description based on the instruction and input.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

EOS_TOKEN: str = "<|eot_id|>"


# ──────────────────────────────────────────────
# Data‑class configs
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for the Mistral‑7B LLM."""

    model_name: str = "unsloth/mistral-7b-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None
    load_in_4bit: bool = True

    # LoRA hyper‑parameters
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3977

    # Paths
    saved_model_dir: str = "saved_mistral_7b_model"


@dataclass
class TrainingConfig:
    """Configuration for the SFTTrainer."""

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 20
    max_steps: int = 150
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.05
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"


@dataclass
class InferenceConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 150
    use_cache: bool = True
    temperature: float = 0.9
    top_p: float = 0.7
    do_sample: bool = True


@dataclass
class ImageGenConfig:
    """Configuration for Stable Diffusion image generation."""

    model_id: str = "stabilityai/stable-diffusion-2-1"
    torch_dtype: str = "float16"
    use_safetensors: bool = True
    num_inference_steps: int = 75
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512


@dataclass
class DataConfig:
    """Configuration for the training dataset."""

    data_file: str = "BOOK_FINAL.csv"
    split: str = "train"
    dataset_text_field: str = "text"
    dataset_num_proc: int = 2
    packing: bool = False

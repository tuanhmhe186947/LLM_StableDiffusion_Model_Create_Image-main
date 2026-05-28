"""
Training module.

Builds and runs the SFTTrainer for fine-tuning the LLM on the
book-cover-description dataset.
"""

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from .config import DataConfig, TrainingConfig


def build_trainer(
    model,
    tokenizer,
    dataset: Dataset,
    training_config: TrainingConfig,
    data_config: DataConfig,
) -> SFTTrainer:
    """Construct an SFTTrainer with the given configurations.

    Args:
        model: The model (with LoRA adapters).
        tokenizer: The tokenizer.
        dataset: The prepared training dataset.
        training_config: Training hyper-parameters.
        data_config: Data-related settings (text field name, etc.).

    Returns:
        An initialised ``SFTTrainer`` instance.
    """
    training_args = TrainingArguments(
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
        learning_rate=training_config.learning_rate,
        fp16=training_config.fp16,
        logging_steps=training_config.logging_steps,
        optim=training_config.optim,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        seed=training_config.seed,
        output_dir=training_config.output_dir,
        report_to=training_config.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=data_config.dataset_text_field,
        max_seq_length=2048,
        dataset_num_proc=data_config.dataset_num_proc,
        packing=data_config.packing,
        args=training_args,
    )
    return trainer


def run_training(trainer: SFTTrainer) -> None:
    """Execute the training loop.

    Args:
        trainer: A configured ``SFTTrainer``.
    """
    result = trainer.train()
    print(f"Training complete — loss: {result.training_loss:.4f}")

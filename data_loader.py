"""
Data loading and formatting module.

Handles loading the CSV dataset and formatting it into the
Alpaca prompt structure expected by the fine-tuning pipeline.
"""

from typing import Dict, List

from datasets import Dataset, load_dataset

from .config import ALPACA_PROMPT, EOS_TOKEN, DataConfig


def formatting_prompts_func(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Format raw dataset rows into Alpaca-style prompt strings.

    Args:
        examples: A batch of rows with ``instruction``, ``input``,
            and ``output`` columns.

    Returns:
        A dictionary with a single ``text`` key containing the
        formatted prompt strings.
    """
    instructions: List[str] = examples["instruction"]
    inputs: List[str] = examples["input"]
    outputs: List[str] = examples["output"]

    texts: List[str] = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        text = ALPACA_PROMPT.format(
            instruction=instruction,
            input=input_text,
            response=output_text,
        ) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}


def load_training_data(config: DataConfig) -> Dataset:
    """Load and prepare the training dataset from a CSV file.

    Args:
        config: Data configuration specifying file path, split, etc.

    Returns:
        A HuggingFace ``Dataset`` with a ``text`` column ready for
        the SFTTrainer.
    """
    dataset: Dataset = load_dataset(
        "csv",
        data_files=config.data_file,
        split=config.split,
    )
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

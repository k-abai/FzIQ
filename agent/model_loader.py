"""
Model loading utilities for FzIQ.
Handles base model, LoRA, and metamodel checkpoint loading.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEEPSEEK_DISTILL_7B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEEPSEEK_DISTILL_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_base_model(
    model_path: str = DEFAULT_MODEL,
    device: str = "auto",
    load_in_4bit: bool = False,
):
    """Load base model and tokenizer."""
    kwargs = {
        "trust_remote_code": True,
        "device_map": device,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded base model: {model_path}")
    return model, tokenizer


def apply_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA configuration to model for fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_metamodel(checkpoint_path: str, base_model_path: str = DEFAULT_MODEL):
    """Load the current metamodel from a checkpoint directory."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}, loading base model")
        return load_base_model(base_model_path)

    model, tokenizer = load_base_model(base_model_path)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    logger.info(f"Loaded metamodel from {checkpoint_path}")
    return model, tokenizer


def save_metamodel(model, tokenizer, save_path: str):
    """Save model weights to a checkpoint directory."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved metamodel to {save_path}")

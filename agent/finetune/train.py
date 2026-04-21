"""
FzIQ fine-tuning script using TRL SFTTrainer.

Usage:
    python agent/finetune/train.py \
        --model_path microsoft/Phi-3-mini-4k-instruct \
        --dataset_path data/training_pairs.json \
        --output_dir metamodel/checkpoints/round_001 \
        --num_epochs 3
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb

from agent.model_loader import load_base_model, apply_lora
from agent.finetune.dataset import FzIQSFTDataset
from agent.finetune.lora_config import FZIQ_LORA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--wandb_project", default="fziq")
    args = parser.parse_args()

    # Init wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    # Load model
    model, tokenizer = load_base_model(args.model_path)
    model = apply_lora(model)

    # Load dataset
    dataset = FzIQSFTDataset.from_json(args.dataset_path, tokenizer, args.max_length)
    logger.info(f"Dataset loaded: {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Fine-tuning complete. Model saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()

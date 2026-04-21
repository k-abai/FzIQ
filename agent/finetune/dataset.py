"""
FzIQ training dataset for SFT fine-tuning.

Converts (scenario, ground_truth) pairs into instruction-following format
for use with HuggingFace TRL SFTTrainer.
"""

import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from agent.prompt_templates import SYSTEM_PROMPT


class FzIQSFTDataset(Dataset):
    """
    Dataset of (scenario, correct_answer) pairs for supervised fine-tuning.
    
    Each item is formatted as a full instruction-following conversation
    including the correct physical reasoning answer.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
    ):
        """
        Args:
            data: list of dicts with keys:
                  - scenario_text: str
                  - ground_truth_outcome: "stable"|"partial_collapse"|"full_collapse"
                  - collapse_sequence: list of block indices
                  - consequence_description: str
            tokenizer: HuggingFace tokenizer
            max_length: max token length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Build target response
        stability = "stable" if item["ground_truth_outcome"] == "stable" else "unstable"
        answer = {
            "stability": stability,
            "confidence": 0.95,
            "consequence": item.get("consequence_description", ""),
            "reasoning": item.get("reasoning", "Physical analysis of the described configuration."),
        }
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["scenario_text"]},
            {"role": "assistant", "content": json.dumps(answer, indent=2)},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @classmethod
    def from_json(cls, path: str, tokenizer, max_length: int = 1024):
        """Load dataset from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(data, tokenizer, max_length)

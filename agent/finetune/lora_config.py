"""LoRA hyperparameter configuration for FzIQ fine-tuning."""

from peft import LoraConfig, TaskType

FZIQ_LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # LoRA rank
    lora_alpha=32,                 # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    inference_mode=False,
)

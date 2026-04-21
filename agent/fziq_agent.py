"""
FzIQ Agent
Loads a Phi-3-mini (or compatible) model, runs inference on block stacking scenarios,
and computes failure gradients for federated aggregation.
"""

import json
import time
import uuid
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from .prompt_templates import format_prompt_string

logger = logging.getLogger(__name__)


@dataclass
class AgentPrediction:
    stability: str           # "stable" | "unstable"
    confidence: float        # 0.0 - 1.0
    consequence: str         # description of what happens
    reasoning: str           # step-by-step reasoning
    raw_response: str        # raw model output
    parse_error: bool = False

    @classmethod
    def from_json(cls, raw: str) -> "AgentPrediction":
        """Parse model JSON output into AgentPrediction."""
        try:
            # Extract JSON from response (model may wrap in markdown)
            text = raw.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            return cls(
                stability=data.get("stability", "unstable"),
                confidence=float(data.get("confidence", 0.5)),
                consequence=data.get("consequence", ""),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}\nRaw: {raw[:200]}")
            return cls(
                stability="unstable",
                confidence=0.0,
                consequence="",
                reasoning="",
                raw_response=raw,
                parse_error=True,
            )


@dataclass
class FailureGradient:
    """Gradient computed from a failure signal, ready for aggregation."""
    gradient: Dict[str, torch.Tensor]
    scenario_hash: str
    agent_id: str
    score: float             # combined human+CNN score 0-1
    timestamp: float = field(default_factory=time.time)
    prediction: Optional[AgentPrediction] = None

    def metadata(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "scenario_hash": self.scenario_hash,
            "score": self.score,
            "timestamp": self.timestamp,
        }


class FzIQAgent:
    """
    FzIQ inference and gradient computation agent.
    
    Runs Phi-3-mini (or any HuggingFace causal LM) with optional LoRA weights.
    Computes failure gradients from graded predictions.
    
    Example:
        agent = FzIQAgent(model_path="microsoft/Phi-3-mini-4k-instruct")
        prediction = agent.predict(scenario)
        # ... after grading ...
        gradient = agent.compute_failure_gradient(prediction, ground_truth, score=0.4)
    """

    def __init__(
        self,
        model_path: str = "microsoft/Phi-3-mini-4k-instruct",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        agent_id: Optional[str] = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.agent_id = agent_id or str(uuid.uuid4())[:8]

        logger.info(f"Loading model from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )

        if lora_path:
            logger.info(f"Loading LoRA weights from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()

    def predict(self, scenario) -> AgentPrediction:
        """
        Run inference on a BlockStackScenario.
        
        Args:
            scenario: BlockStackScenario instance
        Returns:
            AgentPrediction with stability, confidence, consequence, reasoning
        """
        prompt = format_prompt_string(scenario.scenario_text, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return AgentPrediction.from_json(response)

    def compute_failure_gradient(
        self,
        scenario,
        prediction: AgentPrediction,
        ground_truth_outcome: str,   # "stable" | "partial_collapse" | "full_collapse"
        combined_score: float,        # 0-1 from grading pipeline
    ) -> FailureGradient:
        """
        Compute gradient from failure signal.
        
        Higher combined_score → prediction was close → smaller gradient update.
        Lower combined_score → prediction was wrong → larger gradient update.
        
        Args:
            scenario: BlockStackScenario
            prediction: AgentPrediction from self.predict()
            ground_truth_outcome: actual simulation result
            combined_score: 0-1 combined human+CNN grade
        Returns:
            FailureGradient ready for submission to aggregator
        """
        # Enable gradients temporarily for gradient computation
        self.model.train()
        
        # Build target: correct stability label
        stable_gt = ground_truth_outcome == "stable"
        stable_pred = prediction.stability == "stable"

        # Compute cross-entropy loss on stability token
        # Simple formulation: if wrong, loss = 1 - combined_score
        # If right, loss = combined_score * small_penalty (encourage confidence)
        if stable_pred == stable_gt:
            # Prediction direction was correct — small loss proportional to low confidence
            loss_value = combined_score * (1.0 - prediction.confidence) * 0.1
        else:
            # Prediction was wrong — loss proportional to how wrong (inverse of score)
            loss_value = (1.0 - combined_score)

        # Build a simple scalar loss tensor
        prompt = format_prompt_string(scenario.scenario_text, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Forward pass to get loss
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        raw_loss = outputs.loss
        weighted_loss = raw_loss * loss_value

        # Compute gradients over LoRA parameters only
        weighted_loss.backward()
        
        gradient = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient[name] = param.grad.detach().clone().cpu()
                param.grad = None  # zero grad

        self.model.eval()

        return FailureGradient(
            gradient=gradient,
            scenario_hash=scenario.hash(),
            agent_id=self.agent_id,
            score=combined_score,
            prediction=prediction,
        )

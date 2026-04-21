"""
FzIQ CNN Physics Verifier
Trained to scale human judgment by predicting physical plausibility scores.
Input: (scenario_image, prediction_text_embedding, ground_truth_embedding)
Output: plausibility score 0-1
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PhysicsVerifierCNN(nn.Module):
    """
    Multimodal physics plausibility verifier.
    
    Architecture:
    - Visual encoder: ResNet-18 backbone → 256-dim visual features
    - Text encoder: projects concatenated sentence embeddings → 256-dim text features  
    - Fusion: 512-dim → 256 → 64 → 1 (sigmoid) 
    
    Trained on human-graded (scenario, prediction) pairs to approximate human judgment.
    Target: agree with human grade within 1 point on 85%+ of held-out test set.
    """

    def __init__(self, text_embedding_dim: int = 384):
        super().__init__()

        # Visual encoder: ResNet-18 with ImageNet weights
        self.visual_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder.fc = nn.Linear(512, 256)

        # Text projection: prediction + ground_truth embeddings concatenated
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim * 2, 256),
            nn.ReLU(),
        )

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        scenario_img: torch.Tensor,           # (B, 3, H, W)
        prediction_embedding: torch.Tensor,   # (B, 384)
        outcome_embedding: torch.Tensor,      # (B, 384)
    ) -> torch.Tensor:
        """
        Args:
            scenario_img: RGB image of scenario at peak state
            prediction_embedding: sentence-transformer embedding of agent prediction text
            outcome_embedding: sentence-transformer embedding of ground truth outcome text
        Returns:
            Plausibility score tensor (B, 1), values 0-1
        """
        visual_feat = self.visual_encoder(scenario_img)            # (B, 256)
        
        text_input = torch.cat([prediction_embedding, outcome_embedding], dim=-1)  # (B, 768)
        text_feat = self.text_projection(text_input)               # (B, 256)
        
        combined = torch.cat([visual_feat, text_feat], dim=-1)     # (B, 512)
        return self.fusion(combined)                                 # (B, 1)

    def predict_score(
        self,
        scenario_img: torch.Tensor,
        prediction_text: str,
        outcome_text: str,
        sentence_model,
        device: str = "cpu",
    ) -> float:
        """
        Convenience method: compute plausibility score from raw text inputs.
        
        Args:
            scenario_img: preprocessed scenario image tensor
            prediction_text: agent's prediction as a string
            outcome_text: ground truth outcome description
            sentence_model: SentenceTransformer instance for embedding
        Returns:
            float plausibility score 0-1
        """
        self.eval()
        with torch.no_grad():
            pred_emb = torch.tensor(
                sentence_model.encode(prediction_text), dtype=torch.float32
            ).unsqueeze(0).to(device)
            out_emb = torch.tensor(
                sentence_model.encode(outcome_text), dtype=torch.float32
            ).unsqueeze(0).to(device)
            img = scenario_img.unsqueeze(0).to(device)
            score = self.forward(img, pred_emb, out_emb)
        return float(score.squeeze().cpu())

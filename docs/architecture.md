# FzIQ Architecture Deep Dive

## 1. Physics Environment (Isaac Lab)

FzIQ uses NVIDIA Isaac Lab (Isaac Sim 4.5+) to generate physically accurate block stacking scenarios and simulate their outcomes. Key parameters:

- **Blocks:** 2-10 per scenario, mass 0.1-5.0 kg, dimensions 5-30 cm, friction 0.1-0.9
- **Surface:** friction 0.1-0.9, tilt 0-15°  
- **Perturbation:** lateral force 0-20N applied to top block at 0.5s

Each scenario is converted to natural language by the `NaturalLanguageGenerator`, which templates physical parameters into descriptions like:

> *"A stack of 4 blocks sits on a slightly tilted surface (8°). From bottom to top: a heavy 4.1 kg large wood block at the bottom; above it a medium 2.3 kg medium-sized plastic block; a light 1.1 kg small metal block; and a very light 0.3 kg small foam block at the top. A strong lateral force of 12.0 N is applied to the top block from the left..."*

## 2. Agent Model (Phi-3-mini + LoRA)

- **Base model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Fine-tuning:** LoRA rank=16, alpha=32, target modules: q_proj, v_proj
- **Trainable parameters:** ~1% of total (~38M)
- **VRAM:** ~4GB at fp16

The agent outputs structured JSON:
```json
{
  "stability": "unstable",
  "confidence": 0.82,
  "consequence": "The top foam block will slide left immediately...",
  "reasoning": "The applied 12N force exceeds the static friction..."
}
```

## 3. Grading Pipeline

**Phase 1 — Human only (first 2000 grades):**
Combined score = normalized human average (1-5 → 0-1)

**Phase 2 — Human + CNN (2000+ grades):**
```
combined = human_weight * human_normalized + cnn_weight * cnn_score
```
where `cnn_weight` scales from 0.1 → 0.4 as human grades accumulate (0 → 10,000).

**CNN Verifier architecture:**
- Visual encoder: ResNet-18 → 256-dim
- Text encoder: sentence-transformer (all-MiniLM-L6) → project to 256-dim
- Fusion: 512 → 256 → 64 → 1 (sigmoid)

## 4. Failure Gradient Computation

The failure signal is the inverse of the combined score: low score (wrong prediction) → large gradient update.

```python
weighted_loss = raw_loss * (1.0 - combined_score)
```

Only LoRA parameters are updated, keeping the base model frozen and enabling efficient federated aggregation.

## 5. Federated Aggregation

Every 24 hours, the aggregator:
1. Collects all gradients from the buffer
2. Computes per-gradient weight: `w = (1 - score) * agent_reliability`
3. Normalizes weights and computes weighted average gradient
4. Applies update: `param -= lr * aggregated_gradient`
5. Computes SHA-256 of new model parameters
6. Writes `{round, hashes, scores, model_hash_before, model_hash_after}` to opBNB
7. Pushes updated metamodel to all agents

## 6. On-Chain Audit Log

The `FzIQTrainingLog.sol` contract on opBNB stores one record per round. Scores are stored as `uint256` scaled by 1e4 (so 0.7234 becomes 7234). Gas cost per round: ~$0.0001.

Any researcher can:
```python
contract.getRoundDetails(round_id)  # → all agent IDs, scenario hashes, scores
contract.verifyRound(round_id, model_hash)  # → True/False
```

## 7. TD Value Function for Scenario Prioritization

A TD(0) value function tracks which scenario types (by `num_blocks` × `difficulty_decile`) produce the most durable model improvements. After each aggregation round, the benchmark delta serves as the reward signal, updating scenario state values. Future scenario batches are biased toward high-value states.

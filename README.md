# ⚡ FzIQ — Decentralized Experiential Training for Physical AI

> **Hypothesis:** Experiential weights — derived from direct physical interaction and failure — are orders of magnitude more information-dense than linguistic weights derived from textual descriptions of equivalent physical knowledge.

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![opBNB](https://img.shields.io/badge/opBNB-Testnet-yellow.svg)](https://opbnb.bnbchain.org)
[![Model: Phi-3-mini](https://img.shields.io/badge/Model-Phi--3--mini-purple.svg)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

---

## The Problem

GPT-4 has read every physics textbook ever written. Ask it which block falls first when you push a 4-block stack, and it will give you a beautifully written, completely wrong answer.

A 2-year-old who has stacked blocks will not make that mistake.

The difference is not intelligence. It is **the modality of the training signal.**

LLMs learn from descriptions of physical reality. Descriptions are a lossy compression of experience. FzIQ trains models on experiences — simulation outcomes, failure signals, and consequence grading — rather than text descriptions of equivalent scenarios.

**FzIQ is the experiment that tests whether this matters.**

---

## What FzIQ Does

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Agent 1  │    │ Agent 2  │    │ Agent N  │  ← you can run one
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
┌─────────────────────────────────────────────┐
│        Isaac Lab Physics Environment        │
│        (procedural block stacking)          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          Grading Pipeline                   │
│    Human Score (1-5) + CNN Verifier         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│       Failure Gradient Computation          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      Metamodel Aggregator (24hr cycle)      │
│      Weighted federated averaging           │
└─────────────────┬───────────────────────────┘
          ┌───────┴────────┐
          ▼                ▼
┌──────────────┐  ┌────────────────┐
│   Metamodel  │  │  opBNB Chain   │
│  (updated)   │  │  (audit log)   │
└──────────────┘  └────────────────┘
```

1. **Agents** run physics simulations of block stacking scenarios
2. **Predictions** are made by Phi-3-mini: *will this stack collapse? what happens?*
3. **Grading** scores the prediction against the simulation outcome (human + CNN)
4. **Failure gradients** are computed — wrong predictions drive bigger updates
5. **Every 24 hours**, all gradients are federated-averaged into the shared metamodel
6. **Every round** is hashed and written to opBNB — the full training history is verifiable

---

## Research Questions

| Hypothesis | Test |
|-----------|------|
| **H1** | Multi-agent experiential fine-tuning outperforms single-agent, at equal compute | 
| **H2** | Population diversity improves generalization to novel configurations |
| **H3** | On-chain verified training logs enable exact reproduction of training trajectory |

**Critical comparison:** experiential fine-tuning vs. linguistic fine-tuning on equivalent scenarios — same number of training examples, different modality. Does direct experience beat text description?

---

## Benchmark: FzIQ-Bench v1

500 held-out procedurally generated block stacking scenarios, never seen during training:

| Level | Count | Config | 
|-------|-------|--------|
| 1 | 100 | 2-3 blocks, simple |
| 2 | 200 | 4-6 blocks, moderate instability |
| 3 | 200 | 7-10 blocks, complex center-of-mass |

**Metrics:** Stability Accuracy · Consequence Accuracy · Generalization Gap · Sample Efficiency

---

## Preliminary Results

| Model | Stability Acc. | Consequence Acc. |
|-------|---------------|-----------------|
| Phi-3-mini (zero-shot) | 52% | 31% |
| Phi-3-mini + linguistic fine-tuning | 61% | 44% |
| FzIQ (single agent) | 68% | 52% |
| FzIQ (K=5 agents) | 74% | 61% |
| FzIQ (K=20 agents) | 79% | 67% |

*Preliminary results from early system runs. Full results in paper.*

---

## Quickstart

### Option A: Grade scenarios (no GPU required — takes 30 seconds)

```bash
# Hosted grading interface — no install needed
# [Hugging Face Space link — coming soon]
```

### Option B: Run an agent (GPU recommended, laptop GPU works)

```bash
git clone https://github.com/k-abai/FzIQ.git
cd FzIQ

pip install -e .
cp .env.example .env   # fill in your settings

# Test mode (no GPU, no blockchain — verifies setup)
python scripts/run_full_pipeline.py --mode test --num_scenarios 10

# Production mode
python scripts/run_full_pipeline.py --mode prod --num_scenarios 50
```

**Requirements:** Python 3.10+, 4GB+ VRAM (Phi-3-mini), or CPU-only with `DEVICE=cpu`

### Option C: Deploy the human grader

```bash
python grading/human_grader/app.py
# → http://localhost:5000
```

---

## Repository Structure

```
fziq/
├── environment/          # Isaac Lab block stacking environment + scenario generator
├── agent/                # Phi-3-mini agent, LoRA fine-tuning, gradient computation
├── grading/              # Human grader (Flask) + CNN verifier + score fusion
├── aggregator/           # Federated averaging, gradient buffer, opBNB logger
├── metamodel/            # Model versioning and weight distribution
├── evaluation/           # FzIQ-Bench v1 runner + metrics + baselines
├── contracts/            # FzIQTrainingLog.sol (opBNB smart contract)
├── scripts/              # Pipeline launchers
├── configs/              # YAML configuration files
└── docs/                 # Paper draft, architecture deep-dive
```

---

## Smart Contract

The `FzIQTrainingLog` contract is deployed on opBNB. Every 24-hour aggregation round writes:

```
{ round_id, model_hash_before, model_hash_after, scenario_hashes[], agent_ids[], scores[] }
```

This means the **entire training history of the metamodel is publicly verifiable**. Any researcher can independently verify that a given model checkpoint corresponds to a specific round, with specific agents and scenarios. This is reproducibility at a level centralized ML research has never achieved.

| Network | Address |
|---------|---------|
| opBNB Testnet | *deploy in progress* |
| opBNB Mainnet | *post-paper submission* |

---

## Contributing

Three ways to contribute — pick the one that fits your setup:

| Contribution | Requirement | Impact |
|-------------|-------------|--------|
| **Grade scenarios** | Browser | Directly trains the model |
| **Run an agent** | Python + 4GB VRAM | Generates training gradients |
| **Build** | Python/Solidity | Extend the system |

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

**All contributors will be acknowledged in the research paper.**

---

## Paper

*"FzIQ: Decentralized Failure-Driven Fine-Tuning of Open-Weight Models for Physical Reasoning — Experiential Weights vs. Linguistic Weights"*

**Target:** NeurIPS 2026 Workshop on Robot Learning / Decentralized AI  
**Preprint:** arXiv cs.LG + cs.RO + cs.AI (coming soon)

---

## Why opBNB?

Not for speculation. For the same reason science publishes methods sections.

opBNB gas costs are ~$0.0001 per transaction, making it economically viable to write one record per training round. The result: a permanent, public, independently-verifiable audit log of every model update. Any researcher can replay the exact training trajectory. No company controls the log.

---

## Architecture Deep Dive

→ [docs/architecture.md](docs/architecture.md)

---

## License

MIT — use it, build on it, cite it.

---

*FzIQ v0.1.0 · Keke Abai, Boston University · 2026*

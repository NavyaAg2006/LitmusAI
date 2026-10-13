# LitmusAI 🔬

**A fine-tuned LLM judge that evaluates AI-generated responses the way GPT-4 does — at 3B scale, fully open-source, and deployable for free.**

LitmusAI fine-tunes Llama-3.2-3B-Instruct using QLoRA on 8,000 GPT-4-annotated examples from the UltraFeedback dataset, distilling frontier-model evaluation behavior into a lightweight, deployable judge. It achieves **74% pairwise agreement** with GPT-4 on a 1,000-example held-out benchmark — using only 0.28% of the model's parameters as trainable LoRA adapters.

---

## What It Does

LitmusAI provides two evaluation modes:

### 🔵 Score Mode
Paste any instruction and a single AI-generated response. LitmusAI returns a structured scorecard:

| Criterion | Score | Rationale |
|---|---|---|
| Helpfulness | 4/5 | Addresses the question clearly but misses the edge case... |
| Truthfulness | 5/5 | All factual claims are accurate and well-supported... |
| Instruction Following | 3/5 | Partially follows the format but ignores the constraint... |
| Overall | 4/5 | A solid response that covers the core need well... |

### 🟣 Compare Mode
Paste the same instruction with two different responses. LitmusAI scores both independently, renders a side-by-side grouped bar chart, and delivers a calibrated verdict on which response is better and why.

---

## Why This Matters

Evaluating LLM outputs at scale is a core challenge for every team building AI products. GPT-4-as-a-judge is the standard — but it costs money, introduces latency, and creates a dependency on a closed model. LitmusAI demonstrates that a carefully fine-tuned 3B model can replicate GPT-4's evaluation behavior on a narrow task, making automated LLM evaluation accessible and deployable without API costs.

---

## Architecture

```
UltraFeedback (64K GPT-4 annotated examples)
        │
        ▼
   Preprocessing
   ├── extract_example() — maps raw annotations to JSON schema
   ├── 8K train / 1K holdout split (seed=42)
   └── Prompt formatting with Llama-3.2 chat template
        │
        ▼
  QLoRA Fine-tuning (Google Colab T4)
  ├── Base: Llama-3.2-3B-Instruct
  ├── Quantization: 4-bit NF4 via BitsAndBytes
  ├── LoRA: r=16, alpha=32, target: q/k/v/o_proj
  ├── Trainable params: 9.17M / 3.22B (0.28%)
  └── SFTTrainer: 2 epochs, lr=2e-4, effective batch=16
        │
        ▼
   LoRA Adapter (~50MB)
        │
        ▼
  Evaluation on 1K holdout
  ├── MAE per criterion
  ├── Spearman Rank Correlation per criterion
  └── Pairwise Accuracy (overall)
        │
        ▼
  Streamlit App on HuggingFace Spaces (ZeroGPU)
  ├── Score Mode → structured scorecard
  └── Compare Mode → bar chart + verdict
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Fine-tuning | PyTorch, HuggingFace PEFT, TRL (SFTTrainer) |
| Quantization | BitsAndBytes (4-bit NF4, double quantization) |
| Base Model | Llama-3.2-3B-Instruct (Meta) |
| Dataset | UltraFeedback (openbmb) |
| Model Storage | HuggingFace Hub |
| Frontend | Streamlit |
| Hosting | HuggingFace Spaces (ZeroGPU) |
| Evaluation | SciPy (Spearman), NumPy (MAE) |

---

## Project Structure

```
litmusai/
│
├── data/
│   ├── explore_data.ipynb          # UltraFeedback exploration
│   ├── preprocess.ipynb            # Extraction, formatting, splitting
│   ├── train_formatted.json        # 8K formatted training examples
│   └── holdout_formatted.json      # 1K formatted holdout examples
│
├── training/
│   └── train.ipynb                 # QLoRA fine-tuning on Colab
│
├── evaluation/
│   └── evaluate.ipynb              # MAE, Spearman, Pairwise Accuracy
│
├── app/
│   ├── app.py                      # Streamlit frontend
│   └── requirements.txt
│
├── benchmark_results.json          # Stored evaluation metrics
└── README.md
```

---

## Running Locally

**1. Clone the repo:**
```bash
git clone https://github.com/navya/litmusai
cd litmusai
```

**2. Install dependencies:**
```bash
pip install -r app/requirements.txt
```

**3. Run the app:**
```bash
streamlit run app/app.py
```

> ⚠️ Running locally requires a GPU with at least 8GB VRAM to load the model.

---

## Fine-tuning Details

**Dataset:** `openbmb/UltraFeedback` — 8K training examples sampled from 255K total GPT-4-annotated (instruction, response, scorecard) triples. Each example teaches the model to output a structured JSON scorecard with per-criterion scores and rationales.

**Output schema:**
```json
{
  "helpfulness":           {"score": 4, "rationale": "..."},
  "truthfulness":          {"score": 5, "rationale": "..."},
  "instruction_following": {"score": 3, "rationale": "..."},
  "overall":               {"score": 4, "rationale": "..."}
}
```

**Training infrastructure:** Google Colab free tier (T4 GPU, 15GB VRAM). Peak memory usage ~4GB during training.

**LoRA configuration:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## Acknowledgements

- [Meta AI](https://ai.meta.com) for Llama 3.2
- [OpenBMB](https://github.com/OpenBMB/UltraFeedback) for the UltraFeedback dataset
- [HuggingFace](https://huggingface.co) for PEFT, TRL, and Spaces
- [Tim Dettmers et al.](https://arxiv.org/abs/2305.14314) for QLoRA

---

## License

Base model subject to [Meta's Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

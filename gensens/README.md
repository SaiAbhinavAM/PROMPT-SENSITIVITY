# GenSens: Adaptive Prompt Sensitivity Benchmark

GenSens is a research benchmark for studying **prompt sensitivity in complex LLM tasks**. It provides systematically paraphrased prompt variants across four diverse task domains, enabling rigorous measurement of how large language models respond to semantically equivalent but lexically different instructions.

## Dataset Statistics

| Task           | Instances | Variants/Instance | Total Pairs | Source Dataset       |
|----------------|-----------|-------------------|-------------|----------------------|
| Summarization  | 200       | 8                 | 1,600       | CNN/DailyMail        |
| Code           | 200       | 8                 | 1,600       | HumanEval + MBPP     |
| Creative       | 200       | 8                 | 1,600       | WritingPrompts       |
| Dialogue       | 200       | 8                 | 1,600       | MultiWOZ 2.2         |
| **Total**      | **800**   | **8**             | **6,400**   |                      |

**Paraphrase Model:** LLaMA-3-8B-Instruct (bfloat16)
**Similarity Filter:** SBERT (all-mpnet-base-v2), cosine similarity ≥ 0.82
**Strategies:** 16 diverse paraphrase strategies (formal, casual, synonyms, reordered, etc.)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.9
- PyTorch (with CUDA support for H100)
- Transformers, Accelerate, BitsAndBytes
- Datasets (HuggingFace)
- Sentence-Transformers
- NumPy, Pandas, SciPy, tqdm

## Usage

### Generate the full dataset

```bash
python scripts/generate_dataset.py --task all --n_instances 200 --n_variants 8
```

### Generate a single task

```bash
python scripts/generate_dataset.py --task summarization --n_instances 200 --n_variants 8
```

### CLI Arguments

| Argument             | Default | Description                              |
|----------------------|---------|------------------------------------------|
| `--task`             | `all`   | Task to generate (all/summarization/code/creative/dialogue) |
| `--n_instances`      | `200`   | Number of source instances per task      |
| `--n_variants`       | `8`     | Number of paraphrase variants per instance |
| `--seed`             | `42`    | Random seed for reproducibility          |
| `--checkpoint_every` | `20`    | Save checkpoint every N instances        |

### Validate the dataset

```bash
python scripts/validate_dataset.py
```

## Output Format

Each task produces a JSONL file (`data/gensens_<task>_200inst_8var.jsonl`) where each line is a JSON record:

```json
{
  "instance_id": "summ_0001",
  "task": "summarization",
  "base_text": "Summarize the following news article in 3-4 sentences...",
  "base_prompt": "Summarize the following...\n\nArticle:\n...\n\nSummary:",
  "metadata": {
    "article": "...",
    "gold_summary": "...",
    "word_count": 423
  },
  "variants": [
    {
      "variant_idx": 0,
      "paraphrased_text": "Provide a brief summary of the news article below...",
      "full_prompt": "Provide a brief summary...\n\nArticle:\n...\n\nSummary:",
      "sbert_similarity": 0.9234,
      "strategy": "formal_tone"
    }
  ],
  "n_variants_generated": 8,
  "generation_time_s": 12.34
}
```

## Project Structure

```
gensens/
├── scripts/
│   ├── data_loaders.py          # Task-specific data loading
│   ├── paraphrase_generator.py  # LLaMA-3 paraphrase generation + SBERT filtering
│   ├── generate_dataset.py      # Main generation pipeline
│   └── validate_dataset.py      # Post-generation validation
├── data/                        # Generated dataset files
├── logs/                        # Generation logs
├── outputs/                     # Additional outputs
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{gensens2025,
  title     = {Adaptive Prompt Sensitivity in Complex LLM Tasks},
  author    = {TBD},
  journal   = {TBD},
  year      = {2025},
  note      = {Dataset: GenSens v1.0}
}
```

## License

- **Dataset:** CC-BY-4.0
- **Code:** MIT

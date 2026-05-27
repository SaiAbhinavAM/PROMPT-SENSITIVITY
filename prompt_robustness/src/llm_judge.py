import os
import re
import torch
from typing import Dict, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Global judge pipeline initialized once
judge_model = None
judge_tokenizer = None
judge_device = None

# Judge model. Local default: flan-t5-large (seq2seq). On the H100 set
# JUDGE_MODEL to a large independent instruct model (e.g. a 70B-AWQ repo), which
# is routed through the shared causal backend (llm_backend.ChatLLM).
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/flan-t5-large")

# seq2seq (T5/BART/Pegasus) judges use the local pipeline below; everything else
# is treated as a causal instruct model and routed through ChatLLM.
_SEQ2SEQ_KEYS = ("t5", "flan", "bart", "pegasus", "mt5", "led")


def _judge_is_seq2seq() -> bool:
    return any(k in JUDGE_MODEL.lower() for k in _SEQ2SEQ_KEYS)


def _causal_judge_texts(input_text: str, summaries: list) -> List[str]:
    """Get raw text outputs from the causal 70B judge (batched). Returns one string per summary."""
    from .llm_backend import get_chat_llm
    llm = get_chat_llm(JUDGE_MODEL, quantization=os.getenv("JUDGE_QUANTIZATION"))
    msgs = [
        [
            {"role": "system", "content": "You are a strict summary evaluator. "
             "Rate how well the summary captures the article's main facts on a 1-5 "
             "scale (1=very poor, 5=excellent). Reply with ONLY the single digit."},
            {"role": "user", "content": f"Article:\n{input_text[:4000]}\n\nSummary:\n{s}\n\nScore (1-5):"},
        ]
        for s in summaries
    ]
    return llm.chat_batch(msgs, max_new_tokens=4)


def _causal_judge_scores(input_text: str, summaries: list) -> list:
    """Score summaries 1-5 with a large causal instruct judge (batched)."""
    return [parse_judge_output(t) for t in _causal_judge_texts(input_text, summaries)]

def get_judge():
    global judge_model, judge_tokenizer, judge_device
    if judge_model is None:
        if torch.cuda.is_available():
            judge_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            judge_device = torch.device("mps")
        else:
            judge_device = torch.device("cpu")

        print(f"Loading LLM Judge ({JUDGE_MODEL}) on {judge_device}...")
        judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        judge_model = AutoModelForSeq2SeqLM.from_pretrained(JUDGE_MODEL).to(judge_device)
        judge_model.eval()

    return judge_model, judge_tokenizer, judge_device

def parse_judge_output(text: str) -> float:
    match = re.search(r"\d", text)
    if match:
        return int(match.group()) / 5.0
    else:
        print("⚠️ Failed to parse judge score")
        return 0.5

def llm_judge(input_text: str, summary: str, retries: int = 1) -> str:
    model, tokenizer, device = get_judge()
    
    prompt = f"""
You are an expert evaluator.

Evaluate how good the summary is compared to the article.

Score from 1 to 5:

1 = Very poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Return ONLY a single number (1, 2, 3, 4, or 5).

Article:
{input_text}

Summary:
{summary}

Score:
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.0
            )
            
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n=== JUDGE RAW OUTPUT ===")
        print(output)
        print("========================\n")
        
        # Failsafe parsing check
        temp_score = parse_judge_output(output)
        if temp_score == 0.0 or temp_score == 0.5:
            if retries > 0:
                print("⚠️ Invalid judge output, retrying once...")
                return llm_judge(input_text, summary, retries - 1)
            else:
                return "3"
                
        return output
    except Exception as e:
        print(f"Error in judge generation: {e}")
        return ""


def llm_judge_mean(input_text: str, summaries: list) -> float:
    """Judge ALL prompt variants and return the mean normalized score.

    Rectification R5: the human-quality signal should reflect quality across
    every prompt variant, not just responses[0]. Returns a score in [0, 1].
    """
    mean, _ = llm_judge_with_raw(input_text, summaries)
    return mean


def llm_judge_with_raw(
    input_text: str, summaries: list
) -> Tuple[float, List[Dict]]:
    """Judge ALL variants; return (mean_score, per_variant_raw_outputs).

    per_variant_raw_outputs — one dict per valid variant:
      {"raw_output": str, "score": float}   (score is the normalised 0-1 value)

    The raw_output is the exact text the 70B model produced before parsing.
    Stored in responses.csv so results can be audited without re-running the judge.
    """
    valid = [s for s in summaries if (s or "").strip()]
    if not valid:
        return 0.5, []

    raw_outputs: List[Dict] = []
    if _judge_is_seq2seq():
        for s in valid:
            raw_text = llm_judge(input_text, s)
            score = parse_judge_output(raw_text)
            raw_outputs.append({"raw_output": raw_text.strip(), "score": round(score, 4)})
    else:
        texts = _causal_judge_texts(input_text, valid)
        for raw_text in texts:
            score = parse_judge_output(raw_text)
            raw_outputs.append({"raw_output": raw_text.strip(), "score": round(score, 4)})

    scores = [d["score"] for d in raw_outputs]
    mean_score = sum(scores) / len(scores) if scores else 0.5
    return mean_score, raw_outputs

import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Global judge pipeline initialized once
judge_model = None
judge_tokenizer = None
judge_device = None

def get_judge():
    global judge_model, judge_tokenizer, judge_device
    if judge_model is None:
        if torch.cuda.is_available():
            judge_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            judge_device = torch.device("mps")
        else:
            judge_device = torch.device("cpu")
            
        print(f"Loading LLM Judge (google/flan-t5-base) on {judge_device}...")
        judge_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        judge_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(judge_device)
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

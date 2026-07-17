python -c "
from transformers import AutoTokenizer

models = [
'meta-llama/Llama-3.1-8B-Instruct',
'hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
'mistralai/Mistral-7B-Instruct-v0.3',
'Qwen/Qwen2.5-7B-Instruct'
]

for m in models:
    try:
        AutoTokenizer.from_pretrained(m)
        print(f'ACCESS OK: {m}')
    except Exception as e:
        print(f'FAILED: {m}')
        print(e)
"

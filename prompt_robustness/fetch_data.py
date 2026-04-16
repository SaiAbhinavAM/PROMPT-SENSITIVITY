import json
import os
from datasets import load_dataset

def main():
    print("Fetching CNN/DailyMail dataset...")
    # Load the first 50 samples from the train split
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:50]")
    
    out_data = []
    for row in dataset:
        # We truncate the article to 1500 characters so it doesn't overwhelm the local LLaMA model context window or slow down inference too much.
        article = row["article"]
        if len(article) > 1500:
            article = article[:1500] + "..."
            
        out_data.append({
            "input_text": article,
            "reference_output": row["highlights"],
            "topic_label": "news"
        })
        
    data_path = os.path.join("data", "sample_dataset.json")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Successfully saved {len(out_data)} samples to {data_path}!")

if __name__ == "__main__":
    main()

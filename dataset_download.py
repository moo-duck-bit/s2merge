import sys
import os
import pandas as pd
import json
from pathlib import Path

# dataset = load_dataset("openai/gsm8k", "main", cache_dir="./datasets")

def download_multiarith():
    """
    Download MultiArith dataset using Hugging Face datasets and convert to JSON format
    """
    # Import inside function to avoid conflicts
    if 'datasets' in sys.modules:
        local_datasets = sys.modules['datasets']
        if hasattr(local_datasets, '__file__') and 'AgentDropout/datasets' in local_datasets.__file__:
            del sys.modules['datasets']
    
    # Temporarily filter sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if current_dir not in p]
    
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    
    base_path = Path(__file__).parent / "datasets" / "MultiArith"
    
    # Check if data already exists
    if (base_path / "test.json").exists() and (base_path / "train.json").exists():
        print("MultiArith data already exists. Skipping download.")
        return
    
    print("Downloading MultiArith dataset from Hugging Face...")
    
    # Create directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Hugging Face
    dataset = load_dataset("ChilleD/MultiArith", cache_dir="./datasets")
    
    print("Converting dataset to JSON format...")
    
    # Save train and test splits
    for split_name in ["train", "test"]:
        if split_name not in dataset:
            print(f"Warning: {split_name} split not found in dataset")
            continue
        
        split_data = dataset[split_name]
        output_data = []
        
        for item in split_data:
            # Convert to the format expected by multiarith_data_process
            output_data.append({
                'question': item.get('question', ''),
                'answer': item.get('final_ans', ''),
                'chain': item.get('chain', '')
            })
        
        output_path = base_path / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name} split: {len(output_data)} questions saved to {output_path}")
    
    print("\nMultiArith dataset download and conversion complete!")

def download_svamp():
    """
    Download SVAMP dataset using Hugging Face datasets and convert to JSON format
    """
    # Import inside function to avoid conflicts
    if 'datasets' in sys.modules:
        local_datasets = sys.modules['datasets']
        if hasattr(local_datasets, '__file__') and 'AgentDropout/datasets' in local_datasets.__file__:
            del sys.modules['datasets']
    
    # Temporarily filter sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if current_dir not in p]
    
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    
    base_path = Path(__file__).parent / "datasets" / "SVAMP"
    
    # Check if data already exists
    if (base_path / "test.json").exists() and (base_path / "train.json").exists():
        print("SVAMP data already exists. Skipping download.")
        return
    
    print("Downloading SVAMP dataset from Hugging Face...")
    
    # Create directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Hugging Face
    dataset = load_dataset("ChilleD/SVAMP", cache_dir="./datasets")
    
    print("Converting dataset to JSON format...")
    
    # Save train and test splits
    for split_name in ["train", "test"]:
        if split_name not in dataset:
            print(f"Warning: {split_name} split not found in dataset")
            continue
        
        split_data = dataset[split_name]
        output_data = []
        
        for item in split_data:
            # Convert to the format expected by svamp_data_process
            # SVAMP format: Body, Question, Answer
            output_data.append({
                'Body': item.get('Body', ''),
                'Question': item.get('Question', ''),
                'Answer': item.get('Answer', '')
            })
        
        output_path = base_path / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name} split: {len(output_data)} questions saved to {output_path}")
    
    print("\nSVAMP dataset download and conversion complete!")

def download_aqua():
    """
    Download AQUA-RAT dataset using Hugging Face datasets and convert to JSONL format
    """
    # Import inside function to avoid conflicts
    if 'datasets' in sys.modules:
        local_datasets = sys.modules['datasets']
        if hasattr(local_datasets, '__file__') and 'AgentDropout/datasets' in local_datasets.__file__:
            del sys.modules['datasets']
    
    # Temporarily filter sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if current_dir not in p]
    
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    
    base_path = Path(__file__).parent / "datasets" / "aqua"
    
    # Check if data already exists
    if (base_path / "test.jsonl").exists() and (base_path / "val.jsonl").exists():
        print("AQUA data already exists. Skipping download.")
        return
    
    print("Downloading AQUA-RAT dataset from Hugging Face...")
    
    # Create directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Hugging Face
    dataset = load_dataset("deepmind/aqua_rat", cache_dir="./datasets")
    
    print("Converting dataset to JSONL format...")
    
    # Map HuggingFace split names: train, validation, test
    split_mapping = {
        "validation": "val",
        "test": "test"
    }
    
    # Save validation and test splits
    for hf_split, our_split in split_mapping.items():
        if hf_split not in dataset:
            print(f"Warning: {hf_split} split not found in dataset")
            continue
        
        split_data = dataset[hf_split]
        output_path = base_path / f"{our_split}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                # Convert to the format expected by aqua_data_process
                # AQUA format: question, options, rationale, correct (answer)
                output_item = {
                    'question': item.get('question', ''),
                    'options': item.get('options', []),
                    'rationale': item.get('rationale', ''),
                    'correct': item.get('correct', '')
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        
        print(f"{our_split} split: {len(split_data)} questions saved to {output_path}")
    
    print("\nAQUA-RAT dataset download and conversion complete!")

def download_humaneval():
    """
    Download HumanEval dataset using Hugging Face datasets and convert to JSONL format
    """
    # Import inside function to avoid conflicts
    if 'datasets' in sys.modules:
        local_datasets = sys.modules['datasets']
        if hasattr(local_datasets, '__file__') and 'AgentDropout/datasets' in local_datasets.__file__:
            del sys.modules['datasets']
    
    # Temporarily filter sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if current_dir not in p]
    
    try:
        from datasets import load_dataset
    finally:
        sys.path = original_path
    
    base_path = Path(__file__).parent / "datasets" / "humaneval"
    
    # Check if data already exists
    if (base_path / "humaneval-py.jsonl").exists():
        print("HumanEval data already exists. Skipping download.")
        return
    
    print("Downloading HumanEval dataset from Hugging Face...")
    
    # Create directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Hugging Face
    dataset = load_dataset("openai_humaneval", cache_dir="./datasets")
    
    print("Converting dataset to JSONL format...")
    
    # HumanEval has only test split
    if "test" in dataset:
        split_data = dataset["test"]
        output_path = base_path / "humaneval-py.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                # Convert to the format expected by HumanEval processing
                # HumanEval format: task_id, prompt, canonical_solution, test, entry_point
                output_item = {
                    'task_id': item.get('task_id', ''),
                    'prompt': item.get('prompt', ''),
                    'canonical_solution': item.get('canonical_solution', ''),
                    'test': item.get('test', ''),
                    'entry_point': item.get('entry_point', '')
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        
        print(f"HumanEval: {len(split_data)} problems saved to {output_path}")
    
    print("\nHumanEval dataset download and conversion complete!")

def download_mmlu():
    """
    Download MMLU dataset using Hugging Face datasets and convert to CSV format
    """
    # Import inside function to avoid conflicts
    # Remove local datasets module from sys.modules if it exists
    if 'datasets' in sys.modules:
        local_datasets = sys.modules['datasets']
        # Check if it's the local one
        if hasattr(local_datasets, '__file__') and 'AgentDropout/datasets' in local_datasets.__file__:
            del sys.modules['datasets']
    
    # Temporarily filter sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if current_dir not in p]
    
    try:
        # Now import HuggingFace datasets
        from datasets import load_dataset
    finally:
        # Restore original path
        sys.path = original_path
    
    base_path = Path(__file__).parent / "datasets" / "MMLU" / "data"
    
    # Check if data already exists
    if base_path.exists() and any(base_path.iterdir()):
        print("MMLU data already exists. Skipping download.")
        return
    
    print("Downloading MMLU dataset from Hugging Face...")
    
    # Create directories
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "dev").mkdir(exist_ok=True)
    (base_path / "val").mkdir(exist_ok=True)
    (base_path / "test").mkdir(exist_ok=True)
    
    # Download dataset from Hugging Face
    dataset = load_dataset("cais/mmlu", "all", cache_dir="./datasets")
    
    # Map Hugging Face split names to our split names
    split_mapping = {
        "auxiliary_train": "dev",
        "validation": "val",
        "test": "test"
    }
    
    print("Converting dataset to CSV format...")
    
    for hf_split, our_split in split_mapping.items():
        if hf_split not in dataset:
            print(f"Warning: {hf_split} not found in dataset")
            continue
            
        split_data = dataset[hf_split]
        
        # Group by subject (category)
        subjects = {}
        for item in split_data:
            subject = item['subject']
            if subject not in subjects:
                subjects[subject] = []
            
            # Format: question, A, B, C, D, answer
            subjects[subject].append({
                'question': item['question'],
                'A': item['choices'][0],
                'B': item['choices'][1],
                'C': item['choices'][2],
                'D': item['choices'][3],
                'correct_answer': ['A', 'B', 'C', 'D'][item['answer']]
            })
        
        # Save each subject as a separate CSV
        for subject, items in subjects.items():
            df = pd.DataFrame(items)
            output_path = base_path / our_split / f"{subject}_{our_split}.csv"
            df.to_csv(output_path, index=False, header=False)
        
        print(f"{our_split} split: {len(subjects)} subjects, {len(split_data)} questions")
    
    print("\nMMLD dataset download and conversion complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'mmlu', 'multiarith', 'svamp', 'aqua', 'humaneval'],
                        help='Which dataset to download (default: all)')
    args = parser.parse_args()
    
    if args.dataset in ['all', 'mmlu']:
        download_mmlu()
    
    if args.dataset in ['all', 'multiarith']:
        download_multiarith()
    
    if args.dataset in ['all', 'svamp']:
        download_svamp()
    
    if args.dataset in ['all', 'aqua']:
        download_aqua()
    
    if args.dataset in ['all', 'humaneval']:
        download_humaneval()
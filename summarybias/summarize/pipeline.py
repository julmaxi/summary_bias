from transformers import pipeline
from pathlib import Path
import json

import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default="data/onto_nw_all_gender-balanced-10.jsonl")
    parser.add_argument("-o", "--output", type=Path, required=False)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--tokenizer", type=str, required=False)

    args = parser.parse_args()

    Path("summaries").mkdir(exist_ok=True)

    out_path = args.output if args.output is not None else Path("summaries") / (args.input.name + "_" + args.model.replace("/", "-") + "_summaries.jsonl")
    summarizer = pipeline("summarization", model=args.model, tokenizer=args.model if args.tokenizer is None else args.tokenizer, device=0)

    with open(args.input) as f_in, open(out_path, "w") as f_out:
        all_samples = [json.loads(line) for line in f_in]
        summaries = summarizer([s["text"] for s in all_samples], batch_size=4, truncation=True)

        for sample, summary in zip(all_samples, summaries):
            sample["summary"] = summary["summary_text"]
            f_out.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    main()

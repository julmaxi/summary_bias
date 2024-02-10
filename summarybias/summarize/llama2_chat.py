from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, pipeline
from transformers.pipelines import TextGenerationPipeline
import transformers
from pathlib import Path
import json

import argparse
from tqdm import tqdm
import torch
import random

import datasets
from transformers.pipelines.pt_utils import KeyDataset

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def chat_completion(
    dialog
) -> None:
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    dialog = [
        {
            "role": dialog[1]["role"],
            "content": B_SYS
            + dialog[0]["content"]
            + E_SYS
            + dialog[1]["content"],
        }
    ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_tokens: str = ''.join([
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
    ])
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return dialog_tokens


from transformers import Pipeline


class ChatSummarizationPipeline(TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        self.added_bias = kwargs.get("add_bias")
        del kwargs["add_bias"]
        super().__init__(*args, **kwargs)

        assert self.added_bias in [None, "female"]
    
    def preprocess(self, article, prefix="", handle_long_generation=None, **generate_kwargs):
        possible_prompts = [
            "Please summarize the following old text",
            "Please summarize the following old article",
            "Summarize the following old text",
            "Summarize the following old article",
            "Give a summary of the following old text",
            "Give a summary of the following old article",
            "Give me a summary of the following old article",
            "Give me a summary of the following old text",
            "I need a summary of the following old article",
            "I need a summary of the following old text",
        ]

        separator = random.choice(["\n", " ", ": ", ":\n", "\n\n"])
        prompt = random.choice(possible_prompts)
        text = self.tokenizer.decode(self.tokenizer(article, truncation=True, max_length=512).input_ids, skip_special_tokens=True)

        if self.added_bias == "female":
            prompt = f"{prompt}. Please put a particular focus on the women mentioned in the text"

        prompt = chat_completion([{"content": f'{prompt}{separator}{text}', "role": "user"}])
        tokenized = self.tokenizer(prompt, padding=False, add_special_tokens=False, return_tensors=self.framework)
        tokenized["prompt_text"] = prompt

        return tokenized

from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "chat-summarization",
    pipeline_class=ChatSummarizationPipeline,
    pt_model=AutoModelForCausalLM,
)

def flatten(it):
    for x in it:
        for y in x:
            yield y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default="data/onto_nw_all_gender-balanced-10.jsonl")
    parser.add_argument("-o", "--output", type=Path, required=False)
    parser.add_argument("-s", "--model_size", type=str, required=True)
    parser.add_argument("-b", "--bias", type=str, choices=["female"], default=None)

    args = parser.parse_args()

    model_name = f"meta-llama/Llama-2-{args.model_size}-chat-hf"

    if args.bias is not None:
        out_file_name = args.input.name + "_" + model_name.replace("/", "-") + f"_bias_{args.bias}_summaries.jsonl"
    else:
        out_file_name = args.input.name + "_" + model_name.replace("/", "-") + "_summaries.jsonl"
    out_path = args.output if args.output is not None else Path("summaries") / out_file_name
    summarizer = pipeline("chat-summarization", model=model_name, tokenizer=model_name, device_map="auto", torch_dtype=torch.float16, add_bias=args.bias)

    with open(args.input) as f_in:
        all_samples = [json.loads(line) for line in f_in]

    dataset = datasets.Dataset.from_dict({"text": [s["text"] for s in all_samples]})
    
    with open(out_path, "w") as f_out:
        summaries = summarizer(KeyDataset(dataset, "text"), batch_size=1, max_new_tokens=256, return_full_text=False)

        for sample, summary in zip(all_samples, flatten(summaries)):
            summary = summary["generated_text"].strip()
            print(summary)
            sample["summary"] = summary
            f_out.write(json.dumps(sample) + "\n")


def main_old():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default="data/onto_nw_all_gender-balanced-10.jsonl")
    parser.add_argument("-o", "--output", type=Path, required=False)
    parser.add_argument("-s", "--model_size", type=str, required=True)

    args = parser.parse_args()

    model_name = f"meta-llama/Llama-2-{args.model_size}-chat-hf"

    out_path = args.output if args.output is not None else Path("summaries") / (args.input.name + "_" + model_name.replace("/", "-") + "_summaries.jsonl")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.eval()
    model.half()
    model.cuda()
    #pipeline = transformers.pipeline(
    #    "text-generation",
    #    model=model_name,
    #    #torch_dtype=torch.float16,
    #)

    possible_prompts = [
        "Please summarize the following old text",
        "Please summarize the following old article",
        "Summarize the following old text",
        "Summarize the following old article",
        "Give a summary of the following old text",
        "Give a summary of the following old article",
        "Give me a summary of the following old article",
        "Give me a summary of the following old text",
        "I need a summary of the following old article",
        "I need a summary of the following old text",
    ]

    with open(args.input) as f_in:
        all_samples = [json.loads(line) for line in f_in]
    #    summaries = summarizer([s["text"] for s in all_samples], batch_size=4, truncation=True)

    summaries = []

    for sample in all_samples:
        separator = random.choice(["\n", " ", ": ", "\n:", "\n\n"])
        prompt = random.choice(possible_prompts)
        text = tokenizer.decode(tokenizer(sample["text"], truncation=True, max_length=512).input_ids, skip_special_tokens=True)
        prompt = chat_completion([{"content": f'{prompt}{separator}{text}', "role": "user"}])
        tokenized = tokenizer([prompt], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        generation = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256, do_sample=True, num_beams=1)
        generation = generation[0, len(input_ids[0]):]
        seq = tokenizer.batch_decode([generation])[0]
        
        print(f"Result: {seq} <<<")
        summaries.append(seq)

    with open(out_path, "w") as f_out:
        for sample, summary in zip(all_samples, summaries):
            sample["summary"] = summary
            f_out.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    main()

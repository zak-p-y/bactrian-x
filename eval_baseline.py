"""Baseline eval loss for a model without any LoRA adapter.

Reproduces exactly the validation split and tokenisation used by `finetune.py`
(same per-language shuffle seed, same cap, same train_test_split seed), so the
resulting eval loss is directly comparable to the pilot runs' `eval_loss`.

Usage:
    python eval_baseline.py \
        --model_name_or_path Qwen/Qwen2.5-1.5B \
        --lang en,ru --max_samples_per_lang 25000 --val_set_size 1000 \
        --model_max_length 1024 --output_json /kaggle/working/results/baseline.json
"""

import argparse
import json
import logging
import sys
from typing import Optional

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data_path", type=str, default="MBZUAI/Bactrian-X")
    parser.add_argument("--lang", type=str, default="en,ru")
    parser.add_argument("--max_samples_per_lang", type=int, default=25000)
    parser.add_argument("--val_set_size", type=int, default=1000)
    parser.add_argument("--model_max_length", type=int, default=1024)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--prompt_template_name", type=str, default="bactrian")
    parser.add_argument("--template_dir", type=str, default="./templates")
    parser.add_argument("--output_json", type=str, default="results/baseline.json")
    return parser.parse_args()


def build_val_split(data_args):
    """Must stay identical to the split logic in finetune.py."""
    all_dataset = []
    for lang in data_args.lang.split(","):
        lang_dataset = load_dataset(data_args.data_path, lang)["train"]
        if data_args.max_samples_per_lang is not None:
            num_samples = min(data_args.max_samples_per_lang, len(lang_dataset))
            lang_dataset = lang_dataset.shuffle(seed=42).select(range(num_samples))
            logger.info("Using %d samples for language '%s'.", num_samples, lang)
        all_dataset.append(lang_dataset)
    merged = concatenate_datasets(all_dataset)
    train_val = merged.train_test_split(
        test_size=data_args.val_set_size, shuffle=True, seed=42
    )
    return train_val["test"]


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompter = Prompter(args.prompt_template_name, args.template_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="right", use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    max_len = args.model_max_length

    def tokenize(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"]
        )
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        user_prompt_len = len(
            tokenizer(user_prompt, truncation=True, max_length=max_len)["input_ids"]
        )
        tok = tokenizer(
            full_prompt + tokenizer.eos_token, truncation=True, max_length=max_len
        )
        tok["labels"] = [IGNORE_INDEX] * user_prompt_len + tok["input_ids"].copy()[
            user_prompt_len:
        ]
        tok.pop("attention_mask")
        return tok

    val_raw = build_val_split(args)
    val_ds = val_raw.map(
        tokenize, remove_columns=val_raw.column_names, desc="tokenize val data"
    )

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    bs = args.per_device_eval_batch_size
    with torch.no_grad():
        for start in range(0, len(val_ds), bs):
            inputs = collator(
                [val_ds[i] for i in range(start, min(start + bs, len(val_ds)))]
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            n_tok = int((inputs["labels"] != IGNORE_INDEX).sum().item())
            total_loss += out.loss.float().item() * n_tok
            total_tokens += n_tok
            n_batches += 1
            if n_batches % 50 == 0:
                logger.info("batch %d / %d", n_batches, (len(val_ds) + bs - 1) // bs)

    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        "model": args.model_name_or_path,
        "adapter": None,
        "lang": args.lang,
        "max_samples_per_lang": args.max_samples_per_lang,
        "val_set_size": len(val_ds),
        "model_max_length": max_len,
        "eval_loss": mean_loss,
        "perplexity": float(torch.tensor(mean_loss).exp()),
        "eval_tokens": total_tokens,
    }
    logger.info("Baseline eval loss: %.6f (ppl %.4f) over %d tokens",
                mean_loss, result["perplexity"], total_tokens)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

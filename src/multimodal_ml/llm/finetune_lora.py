from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a personal LLM with LoRA")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--init_adapter_dir", type=Path, default=None, help="Continue training from existing LoRA adapter")
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/personal_llm_lora"))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    return parser.parse_args()


def format_example(instruction: str, response: str) -> tuple[str, str]:
    prompt = (
        "You are Bharat's personal math and physics assistant. "
        "Show concise steps and formulas.\n"
        f"Question: {instruction}\n"
        "Answer:\n"
    )
    answer = f"{response}\n"
    return prompt, answer


def build_tokenized_sample(tokenizer: AutoTokenizer, prompt: str, answer: str, max_length: int) -> dict:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = (prompt_ids + answer_ids + eos)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids + eos)[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = str(args.init_adapter_dir) if args.init_adapter_dir else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": str(args.train_file)}
    if args.val_file is not None:
        data_files["validation"] = str(args.val_file)

    ds = load_dataset("json", data_files=data_files)

    def preprocess(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for instruction, response in zip(batch["instruction"], batch["response"]):
            prompt, answer = format_example(instruction, response)
            sample = build_tokenized_sample(tokenizer, prompt, answer, args.max_length)
            out["input_ids"].append(sample["input_ids"])
            out["attention_mask"].append(sample["attention_mask"])
            out["labels"].append(sample["labels"])
        return out

    remove_cols = ds["train"].column_names
    ds = ds.map(preprocess, batched=True, remove_columns=remove_cols)

    if args.init_adapter_dir is not None:
        peft_cfg = PeftConfig.from_pretrained(str(args.init_adapter_dir))
        base_model_name = peft_cfg.base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, str(args.init_adapter_dir), is_trainable=True)
        print(f"loaded_init_adapter={args.init_adapter_dir}")
        print(f"init_base_model={base_model_name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    has_validation = "validation" in ds
    train_args_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        bf16=False,
        fp16=False,
    )

    if has_validation:
        train_args_kwargs["load_best_model_at_end"] = True
        train_args_kwargs["metric_for_best_model"] = "eval_loss"
        train_args_kwargs["greater_is_better"] = False

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in init_params:
        train_args_kwargs["evaluation_strategy"] = "epoch" if has_validation else "no"
    else:
        train_args_kwargs["eval_strategy"] = "epoch" if has_validation else "no"

    train_args = TrainingArguments(**train_args_kwargs)

    callbacks = []
    if has_validation:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved_lora_adapter={args.output_dir}")


if __name__ == "__main__":
    main()

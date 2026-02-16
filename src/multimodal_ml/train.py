from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import MultitaskMultimodalDataset, build_vocab, get_train_text_files
from .model import MultiTaskMultimodalModel


TASKS = ["emotion", "command", "product"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask multimodal classifier")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--emotion_classes", type=int, required=True)
    parser.add_argument("--command_classes", type=int, required=True)
    parser.add_argument("--product_classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = {task: 0 for task in TASKS}

    with torch.no_grad():
        for audio, tokens, images, labels in loader:
            audio = audio.to(device)
            tokens = tokens.to(device)
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            logits = model(audio, tokens, images)
            losses = [criterion(logits[task], labels[task]) for task in TASKS]
            loss = sum(losses)

            total_loss += loss.item() * audio.size(0)
            total_count += audio.size(0)

            for task in TASKS:
                preds = logits[task].argmax(dim=1)
                total_correct[task] += (preds == labels[task]).sum().item()

    avg_loss = total_loss / total_count
    acc = {task: total_correct[task] / total_count for task in TASKS}
    mean_acc = sum(acc.values()) / len(TASKS)
    return avg_loss, acc, mean_acc


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_text_files = get_train_text_files(args.data_dir)
    vocab = build_vocab(train_text_files)

    train_ds = MultitaskMultimodalDataset(
        split_dir=args.data_dir / "train",
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        audio_seconds=cfg.audio_seconds,
        max_tokens=cfg.max_tokens,
        image_size=cfg.image_size,
    )
    val_ds = MultitaskMultimodalDataset(
        split_dir=args.data_dir / "val",
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        audio_seconds=cfg.audio_seconds,
        max_tokens=cfg.max_tokens,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MultiTaskMultimodalModel(
        vocab_size=vocab.size,
        num_emotions=args.emotion_classes,
        num_commands=args.command_classes,
        num_products=args.product_classes,
        audio_dim=cfg.audio_embed_dim,
        text_dim=cfg.text_embed_dim,
        image_dim=cfg.image_embed_dim,
        hidden_dim=cfg.fusion_hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_mean_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        running_correct = {task: 0 for task in TASKS}

        for audio, tokens, images, labels in train_loader:
            audio = audio.to(device)
            tokens = tokens.to(device)
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}

            optimizer.zero_grad(set_to_none=True)
            logits = model(audio, tokens, images)
            losses = [criterion(logits[task], labels[task]) for task in TASKS]
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * audio.size(0)
            running_count += audio.size(0)

            for task in TASKS:
                preds = logits[task].argmax(dim=1)
                running_correct[task] += (preds == labels[task]).sum().item()

        train_loss = running_loss / running_count
        train_acc = {task: running_correct[task] / running_count for task in TASKS}
        train_mean_acc = sum(train_acc.values()) / len(TASKS)

        val_loss, val_acc, val_mean_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_mean_acc={train_mean_acc:.4f} "
            f"val_loss={val_loss:.4f} val_mean_acc={val_mean_acc:.4f} "
            f"val_emotion_acc={val_acc['emotion']:.4f} val_command_acc={val_acc['command']:.4f} "
            f"val_product_acc={val_acc['product']:.4f}"
        )

        if val_mean_acc > best_mean_val_acc:
            best_mean_val_acc = val_mean_acc
            ckpt_path = args.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": {"stoi": vocab.stoi, "itos": vocab.itos},
                    "config": cfg.__dict__,
                    "num_emotions": args.emotion_classes,
                    "num_commands": args.command_classes,
                    "num_products": args.product_classes,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

    vocab_path = args.checkpoint_dir / "vocab.json"
    vocab_path.write_text(json.dumps({"stoi": vocab.stoi, "itos": vocab.itos}, indent=2), encoding="utf-8")
    print(f"saved vocab: {vocab_path}")


if __name__ == "__main__":
    main()

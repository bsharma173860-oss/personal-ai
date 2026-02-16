from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


EMOTION_TOKENS = ["happy", "sad", "angry", "calm", "surprised"]
COMMAND_TOKENS = ["start", "stop", "left", "right", "select"]
PRODUCT_TOKENS = ["book", "phone", "laptop", "bottle", "pen"]
LANG_SNIPPETS = [
    "solve integral",
    "resolver ecuacion",
    "resoudre derivée",
    "losung physik",
    "calcola limite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dummy multitask multimodal dataset")
    parser.add_argument("--data_dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--emotion_classes", type=int, default=5)
    parser.add_argument("--command_classes", type=int, default=5)
    parser.add_argument("--product_classes", type=int, default=5)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=2.0)
    return parser.parse_args()


def synth_audio(emotion_id: int, command_id: int, sample_rate: int, seconds: float) -> torch.Tensor:
    n = int(sample_rate * seconds)
    t = torch.linspace(0, seconds, steps=n)
    base_freq = 180 + (emotion_id * 40) + (command_id * 20)
    tone = 0.3 * torch.sin(2 * math.pi * base_freq * t)
    noise = 0.02 * torch.randn_like(tone)
    wav = (tone + noise).unsqueeze(0)
    return wav


def save_wav_pcm16(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    mono = wav.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
    pcm16 = (mono * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def synth_text(emotion_id: int, command_id: int, product_id: int) -> str:
    a = EMOTION_TOKENS[emotion_id % len(EMOTION_TOKENS)]
    b = COMMAND_TOKENS[command_id % len(COMMAND_TOKENS)]
    c = PRODUCT_TOKENS[product_id % len(PRODUCT_TOKENS)]
    lang = random.choice(LANG_SNIPPETS)
    return f"emotion {a} command {b} product {c} math physics {lang}"


def synth_image(product_id: int, emotion_id: int, size: int = 128) -> Image.Image:
    colors = [(220, 60, 60), (60, 180, 80), (70, 110, 220), (220, 180, 60), (160, 70, 200)]
    color = colors[product_id % len(colors)]
    img = Image.new("RGB", (size, size), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    margin = 8 + (emotion_id * 4)
    draw.rectangle([margin, margin, size - margin, size - margin], outline=color, width=6)
    return img


def write_split(
    base_dir: Path,
    split: str,
    count: int,
    emotion_classes: int,
    command_classes: int,
    product_classes: int,
    sample_rate: int,
    seconds: float,
) -> None:
    split_dir = base_dir / split
    for sub in ["audio", "text", "image", "labels_emotion", "labels_command", "labels_product"]:
        (split_dir / sub).mkdir(parents=True, exist_ok=True)

    for i in range(count):
        sample_id = f"{split}_{i:04d}"
        emotion_id = random.randrange(0, emotion_classes)
        command_id = random.randrange(0, command_classes)
        product_id = random.randrange(0, product_classes)

        wav = synth_audio(emotion_id, command_id, sample_rate, seconds)
        save_wav_pcm16(split_dir / "audio" / f"{sample_id}.wav", wav, sample_rate)

        (split_dir / "text" / f"{sample_id}.txt").write_text(
            synth_text(emotion_id, command_id, product_id),
            encoding="utf-8",
        )

        img = synth_image(product_id, emotion_id)
        img.save(split_dir / "image" / f"{sample_id}.jpg")

        (split_dir / "labels_emotion" / f"{sample_id}.txt").write_text(str(emotion_id), encoding="utf-8")
        (split_dir / "labels_command" / f"{sample_id}.txt").write_text(str(command_id), encoding="utf-8")
        (split_dir / "labels_product" / f"{sample_id}.txt").write_text(str(product_id), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(42)

    write_split(
        args.data_dir,
        "train",
        args.train,
        args.emotion_classes,
        args.command_classes,
        args.product_classes,
        args.sample_rate,
        args.seconds,
    )
    write_split(
        args.data_dir,
        "val",
        args.val,
        args.emotion_classes,
        args.command_classes,
        args.product_classes,
        args.sample_rate,
        args.seconds,
    )
    write_split(
        args.data_dir,
        "test",
        args.test,
        args.emotion_classes,
        args.command_classes,
        args.product_classes,
        args.sample_rate,
        args.seconds,
    )

    print(f"dummy dataset created at {args.data_dir}")


if __name__ == "__main__":
    main()

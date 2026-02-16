from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .dataset import Vocab, encode_text, load_audio_mel
from .model import MultiTaskMultimodalModel


TASKS = ["emotion", "command", "product"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with multitask multimodal classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio_path", type=Path, required=True)
    parser.add_argument("--text_path", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--creator_profile", type=Path, default=Path("creator_profile.json"))
    return parser.parse_args()


def load_creator_profile(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = load_creator_profile(args.creator_profile)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint["config"]
    vocab = Vocab(
        stoi=checkpoint["vocab"]["stoi"],
        itos=checkpoint["vocab"]["itos"],
    )

    model = MultiTaskMultimodalModel(
        vocab_size=len(vocab.itos),
        num_emotions=checkpoint["num_emotions"],
        num_commands=checkpoint["num_commands"],
        num_products=checkpoint["num_products"],
        audio_dim=cfg["audio_embed_dim"],
        text_dim=cfg["text_embed_dim"],
        image_dim=cfg["image_embed_dim"],
        hidden_dim=cfg["fusion_hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    audio = load_audio_mel(
        audio_path=args.audio_path,
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        audio_seconds=cfg["audio_seconds"],
    ).unsqueeze(0).to(device)

    text = args.text_path.read_text(encoding="utf-8")
    tokens = encode_text(text, vocab, cfg["max_tokens"]).unsqueeze(0).to(device)

    image_transform = transforms.Compose(
        [
            transforms.Resize((cfg["image_size"], cfg["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(args.image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        creator_name = profile.get("creator_name")
        preferred_greeting = profile.get("preferred_greeting", "Hello")
        priority_domains = profile.get("priority_domains", [])
        if creator_name:
            print(f"{preferred_greeting}, {creator_name}.")
        if priority_domains:
            print(f"priority_domains={','.join(priority_domains)}")

        logits = model(audio, tokens, image)
        for task in TASKS:
            probs = torch.softmax(logits[task], dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred].item())
            print(f"{task}_pred={pred} {task}_confidence={conf:.4f}")


if __name__ == "__main__":
    main()

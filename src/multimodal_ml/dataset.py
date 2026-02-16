from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import wave

import numpy as np
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def size(self) -> int:
        return len(self.itos)


def tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def build_vocab(text_files: List[Path], min_freq: int = 1) -> Vocab:
    freq: Dict[str, int] = {}
    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8")
        for tok in tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1

    itos = ["<pad>", "<unk>"]
    for tok, count in sorted(freq.items()):
        if count >= min_freq:
            itos.append(tok)
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def encode_text(text: str, vocab: Vocab, max_tokens: int) -> torch.Tensor:
    tokens = tokenize(text)
    ids = [vocab.stoi.get(tok, vocab.stoi["<unk>"]) for tok in tokens[:max_tokens]]
    if len(ids) < max_tokens:
        ids.extend([vocab.stoi["<pad>"]] * (max_tokens - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def load_wav_pcm(audio_path: Path) -> Tuple[torch.Tensor, int]:
    with wave.open(str(audio_path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    wav = torch.from_numpy(audio).unsqueeze(0)
    return wav, sample_rate


def resample_1d(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return wav
    target_len = int(round(wav.shape[1] * (dst_sr / src_sr)))
    wav_rs = torch.nn.functional.interpolate(
        wav.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return wav_rs.squeeze(0)


def load_audio_mel(audio_path: Path, sample_rate: int, n_mels: int, audio_seconds: float) -> torch.Tensor:
    wav, sr = load_wav_pcm(audio_path)

    if sr != sample_rate:
        wav = resample_1d(wav, sr, sample_rate)

    target_len = int(sample_rate * audio_seconds)
    if wav.shape[1] < target_len:
        pad = target_len - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    else:
        wav = wav[:, :target_len]

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )(wav)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    mean = mel_spec.mean()
    std = mel_spec.std().clamp_min(1e-6)
    mel_spec = (mel_spec - mean) / std
    return mel_spec


class MultitaskMultimodalDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        vocab: Vocab,
        sample_rate: int,
        n_mels: int,
        audio_seconds: float,
        max_tokens: int,
        image_size: int,
    ):
        self.split_dir = split_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.audio_seconds = audio_seconds
        self.max_tokens = max_tokens

        self.audio_dir = split_dir / "audio"
        self.text_dir = split_dir / "text"
        self.image_dir = split_dir / "image"
        self.labels_emotion_dir = split_dir / "labels_emotion"
        self.labels_command_dir = split_dir / "labels_command"
        self.labels_product_dir = split_dir / "labels_product"

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.sample_ids = self._collect_sample_ids()

    def _collect_sample_ids(self) -> List[str]:
        ids = [p.stem for p in sorted(self.labels_emotion_dir.glob("*.txt"))]
        if not ids:
            raise ValueError(f"No emotion labels found in {self.labels_emotion_dir}")
        return ids

    def _find_image_path(self, sample_id: str) -> Path:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = self.image_dir / f"{sample_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for sample_id={sample_id}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]

        audio_path = self.audio_dir / f"{sample_id}.wav"
        text_path = self.text_dir / f"{sample_id}.txt"
        image_path = self._find_image_path(sample_id)
        emotion_path = self.labels_emotion_dir / f"{sample_id}.txt"
        command_path = self.labels_command_dir / f"{sample_id}.txt"
        product_path = self.labels_product_dir / f"{sample_id}.txt"

        for p in [audio_path, text_path, emotion_path, command_path, product_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        audio = load_audio_mel(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            audio_seconds=self.audio_seconds,
        )

        text = text_path.read_text(encoding="utf-8")
        tokens = encode_text(text, self.vocab, self.max_tokens)

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        labels = {
            "emotion": torch.tensor(int(emotion_path.read_text(encoding="utf-8").strip()), dtype=torch.long),
            "command": torch.tensor(int(command_path.read_text(encoding="utf-8").strip()), dtype=torch.long),
            "product": torch.tensor(int(product_path.read_text(encoding="utf-8").strip()), dtype=torch.long),
        }

        return audio, tokens, image, labels


def get_train_text_files(data_dir: Path) -> List[Path]:
    train_text_dir = data_dir / "train" / "text"
    files = sorted(train_text_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No training text files found in {train_text_dir}")
    return files

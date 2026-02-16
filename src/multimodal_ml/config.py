from dataclasses import dataclass


@dataclass
class TrainConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    audio_seconds: float = 4.0
    max_tokens: int = 32
    text_embed_dim: int = 128
    audio_embed_dim: int = 128
    image_embed_dim: int = 128
    fusion_hidden_dim: int = 256
    image_size: int = 128
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10

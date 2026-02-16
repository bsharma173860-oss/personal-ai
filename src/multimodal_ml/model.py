import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.proj(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, out_dim, batch_first=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        _, h = self.rnn(x)
        return h[-1]


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).flatten(1)
        return self.proj(x)


class MultiTaskMultimodalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_emotions: int,
        num_commands: int,
        num_products: int,
        audio_dim: int = 128,
        text_dim: int = 128,
        image_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder(out_dim=audio_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=text_dim, out_dim=text_dim)
        self.image_encoder = ImageEncoder(out_dim=image_dim)

        total_dim = audio_dim + text_dim + image_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.command_head = nn.Linear(hidden_dim, num_commands)
        self.product_head = nn.Linear(hidden_dim, num_products)

    def forward(self, audio: torch.Tensor, tokens: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.audio_encoder(audio)
        t = self.text_encoder(tokens)
        i = self.image_encoder(image)
        x = torch.cat([a, t, i], dim=1)
        h = self.fusion(x)

        return {
            "emotion": self.emotion_head(h),
            "command": self.command_head(h),
            "product": self.product_head(h),
        }

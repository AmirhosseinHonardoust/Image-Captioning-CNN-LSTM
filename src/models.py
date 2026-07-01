"""Model definitions for CNN-LSTM image captioning."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    """ResNet-50 image encoder.

    The ResNet backbone is frozen by default and the trainable projection maps the
    2048-dimensional pooled CNN feature to the decoder embedding dimension.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        train_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.train_backbone = train_backbone

        from torchvision import models

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on network/cache state
            if not pretrained:
                raise
            warnings.warn(
                "Could not load pretrained ResNet-50 weights; falling back to "
                f"random initialization. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            backbone = models.resnet50(weights=None)

        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(backbone.fc.in_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.train_backbone:
            feats = self.cnn(images).flatten(1)
        else:
            with torch.no_grad():
                feats = self.cnn(images).flatten(1)

        feats = self.fc(feats)
        feats = self.norm(feats)
        return torch.relu(feats)

    def trainable_parameters(self):
        """Return only parameters that should be optimized."""
        return [param for param in self.parameters() if param.requires_grad]


class DecoderLSTM(nn.Module):
    """LSTM caption decoder.

    During training, ``forward`` prepends the image feature as an initial LSTM
    input. The first logit corresponds to that image feature and should be
    ignored for loss computation; logits from position 1 onward predict
    ``caption[:, 1:]``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Return logits for ``[image_feature] + caption_tokens`` inputs."""
        embeddings = self.input_dropout(self.embed(captions))
        features = features.unsqueeze(1)
        inputs = torch.cat([features, embeddings], dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)

    def sample(
        self,
        features: torch.Tensor,
        max_len: int = 20,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Greedy caption generation.

        The image feature first warms up the LSTM state. Generation then starts
        from ``<bos>`` so training and inference are aligned.
        """
        batch_size = features.size(0)
        device = features.device
        outputs = []

        _, states = self.lstm(features.unsqueeze(1))
        next_ids = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            inputs = self.embed(next_ids).unsqueeze(1)
            out, states = self.lstm(inputs, states)
            logits = self.fc(out[:, -1, :])
            next_ids = torch.argmax(logits, dim=1)
            outputs.append(next_ids)

            if (next_ids == eos_id).all():
                break

        if not outputs:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(outputs, dim=1)

    def beam_search(
        self,
        features: torch.Tensor,
        max_len: int = 20,
        bos_id: int = 1,
        eos_id: int = 2,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        """Beam-search decoding for a single image.

        Returns a tensor containing the best generated token ids, excluding
        ``<bos>`` and including ``<eos>`` if it was generated.
        """
        if features.size(0) != 1:
            raise ValueError("beam_search currently supports batch_size=1")
        if beam_size <= 1:
            return self.sample(features, max_len=max_len, bos_id=bos_id, eos_id=eos_id)

        _, initial_states = self.lstm(features.unsqueeze(1))
        beams: list[tuple[list[int], float, Any, bool]] = [([bos_id], 0.0, initial_states, False)]

        for _ in range(max_len):
            candidates: list[tuple[list[int], float, Any, bool]] = []
            for seq, score, states, done in beams:
                if done:
                    candidates.append((seq, score, states, done))
                    continue

                last_id = torch.tensor([seq[-1]], dtype=torch.long, device=features.device)
                inputs = self.embed(last_id).unsqueeze(1)
                out, next_states = self.lstm(inputs, states)
                log_probs = torch.log_softmax(self.fc(out[:, -1, :]), dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_size, dim=-1)

                for log_prob, token_id in zip(top_scores[0], top_ids[0], strict=False):
                    token = int(token_id.item())
                    candidates.append(
                        (
                            seq + [token],
                            score + float(log_prob.item()),
                            next_states,
                            token == eos_id,
                        )
                    )

            def normalized(candidate):
                seq, score, _, _ = candidate
                length = max(1, len(seq) - 1)
                return score / (length**length_penalty)

            beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
            if all(done for _, _, _, done in beams):
                break

        best_seq = beams[0][0][1:]  # remove <bos>
        return torch.tensor(best_seq, dtype=torch.long, device=features.device).unsqueeze(0)


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over spatial image features."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(
        self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, alpha).

        ``encoder_out`` is ``(B, num_pixels, encoder_dim)`` and ``decoder_hidden``
        is ``(B, decoder_dim)``. ``alpha`` is a ``(B, num_pixels)`` distribution
        over spatial locations; ``context`` is the ``(B, encoder_dim)`` weighted sum.
        """
        att1 = self.enc_att(encoder_out)
        att2 = self.dec_att(decoder_hidden).unsqueeze(1)
        scores = self.full_att(torch.tanh(att1 + att2)).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha


class EncoderCNNAttention(nn.Module):
    """ResNet-50 encoder that keeps the spatial feature grid for attention.

    Unlike :class:`EncoderCNN`, this returns ``(B, num_pixels, encoder_dim)``
    (the un-pooled convolutional map) so the decoder can attend over locations.
    """

    def __init__(
        self,
        pretrained: bool = True,
        train_backbone: bool = False,
        encoded_image_size: int = 7,
    ) -> None:
        super().__init__()
        self.train_backbone = train_backbone
        self.encoder_dim = 2048

        from torchvision import models

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on network/cache state
            if not pretrained:
                raise
            warnings.warn(
                "Could not load pretrained ResNet-50 weights; falling back to "
                f"random initialization. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            backbone = models.resnet50(weights=None)

        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Drop the global average pool and fc head; keep the conv feature map.
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.train_backbone:
            feats = self.cnn(images)
        else:
            with torch.no_grad():
                feats = self.cnn(images)
        feats = self.adaptive_pool(feats)  # (B, C, S, S)
        batch, channels, height, width = feats.shape
        # (B, num_pixels, C)
        return feats.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

    def trainable_parameters(self):
        """Return only parameters that should be optimized."""
        return [param for param in self.parameters() if param.requires_grad]


class AttentionDecoderLSTM(nn.Module):
    """LSTM decoder with visual attention (Show, Attend and Tell).

    ``forward`` returns ``(logits, alphas)``. The decoder is initialized from the
    mean encoder feature and, at each step, attends over the spatial grid to build
    a context vector that is concatenated with the token embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_dim: int = 2048,
        attention_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def _step(
        self, token_embed: torch.Tensor, encoder_out: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, alpha = self.attention(encoder_out, h)
        context = torch.sigmoid(self.f_beta(h)) * context
        h, c = self.lstm_cell(torch.cat([token_embed, context], dim=1), (h, c))
        return h, c, alpha

    def forward(
        self, encoder_out: torch.Tensor, captions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced decode. ``logits[:, t]`` predicts ``captions[:, t]``'s next token."""
        h, c = self.init_hidden(encoder_out)
        embeddings = self.dropout(self.embed(captions))
        logits, alphas = [], []
        for t in range(captions.size(1)):
            h, c, alpha = self._step(embeddings[:, t, :], encoder_out, h, c)
            logits.append(self.fc(self.dropout(h)))
            alphas.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)

    def sample(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 20,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Greedy caption generation from spatial encoder features."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        h, c = self.init_hidden(encoder_out)
        next_ids = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        outputs = []
        for _ in range(max_len):
            h, c, _ = self._step(self.embed(next_ids), encoder_out, h, c)
            next_ids = torch.argmax(self.fc(h), dim=1)
            outputs.append(next_ids)
            if (next_ids == eos_id).all():
                break
        if not outputs:
            return torch.empty(batch_size, 0, dtype=torch.long, device=device)
        return torch.stack(outputs, dim=1)

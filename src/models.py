"""Model definitions for CNN-LSTM image captioning."""

from __future__ import annotations

import warnings

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
        beams = [([bos_id], 0.0, initial_states, False)]

        for _ in range(max_len):
            candidates = []
            for seq, score, states, done in beams:
                if done:
                    candidates.append((seq, score, states, done))
                    continue

                last_id = torch.tensor([seq[-1]], dtype=torch.long, device=features.device)
                inputs = self.embed(last_id).unsqueeze(1)
                out, next_states = self.lstm(inputs, states)
                log_probs = torch.log_softmax(self.fc(out[:, -1, :]), dim=-1)
                top_scores, top_ids = torch.topk(log_probs, beam_size, dim=-1)

                for log_prob, token_id in zip(top_scores[0], top_ids[0]):
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
                return score / (length ** length_penalty)

            beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
            if all(done for _, _, _, done in beams):
                break

        best_seq = beams[0][0][1:]  # remove <bos>
        return torch.tensor(best_seq, dtype=torch.long, device=features.device).unsqueeze(0)

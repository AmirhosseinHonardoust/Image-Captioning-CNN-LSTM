"""Utility classes and functions for image-captioning experiments."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import nltk
import torch
from PIL import Image
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text.

    Uses NLTK when tokenizer resources are installed and falls back to a simple
    whitespace tokenizer for smoke tests or fresh environments.
    """
    cleaned = re.sub(r"[^A-Za-z0-9' ]", " ", str(text).lower())
    try:
        return nltk.word_tokenize(cleaned)
    except LookupError:
        return cleaned.split()


class Vocabulary:
    def __init__(self, min_freq: int = 3) -> None:
        self.min_freq = min_freq
        self.word2id = {PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID, UNK: UNK_ID}
        self.id2word = {PAD_ID: PAD, BOS_ID: BOS, EOS_ID: EOS, UNK_ID: UNK}

    def build(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        for word, count in counter.items():
            if count >= self.min_freq and word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, text: str, max_len: int = 20) -> list[int]:
        token_ids = [self.word2id.get(word, UNK_ID) for word in tokenize(text)][:max_len]
        return [BOS_ID] + token_ids + [EOS_ID]

    def decode(self, ids: Iterable[int]) -> str:
        words = []
        for idx in ids:
            word = self.id2word.get(int(idx), UNK)
            if word in {PAD, BOS}:
                continue
            if word == EOS:
                break
            words.append(word)
        return " ".join(words)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "word2id": self.word2id}, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> Vocabulary:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        vocab = cls(min_freq=obj.get("min_freq", 3))
        vocab.word2id = {str(word): int(idx) for word, idx in obj["word2id"].items()}
        vocab.id2word = {idx: word for word, idx in vocab.word2id.items()}
        return vocab


class CaptionDataset(Dataset):
    def __init__(
        self,
        df,
        images_root: str | Path,
        vocab: Vocabulary,
        split: str = "train",
        max_len: int = 20,
        image_size: int = 224,
    ) -> None:
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_root = Path(images_root)
        self.vocab = vocab
        self.max_len = max_len
        from torchvision import transforms

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        ids = self.vocab.encode(row["caption"], max_len=self.max_len)
        return img, torch.tensor(ids, dtype=torch.long), idx


def pad_collate(batch):
    imgs, seqs, indices = zip(*batch, strict=False)
    imgs = torch.stack(imgs, dim=0)
    lengths = [len(seq) for seq in seqs]
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : len(seq)] = seq
    return (
        imgs,
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
    )


def _normalize_references(refs: Sequence[str] | Sequence[Sequence[str]]) -> list[list[list[str]]]:
    """Convert single-reference or multi-reference text into BLEU input format.

    ``nltk.translate.bleu_score.corpus_bleu`` expects references as
    ``list[list[list[str]]]``: examples -> reference captions -> tokens.
    This helper accepts both ``["caption"]`` and ``[["caption 1", "caption 2"]]``.
    """
    normalized = []
    for ref_group in refs:
        group = [ref_group] if isinstance(ref_group, str) else list(ref_group)
        tokenized_group = [tokenize(ref) for ref in group if str(ref).strip()]
        normalized.append(tokenized_group or [[]])
    return normalized


def compute_bleu(
    gens: Sequence[str], refs: Sequence[str] | Sequence[Sequence[str]], n: int = 4
) -> float:
    """Compute corpus BLEU for generated captions.

    ``refs`` can contain either one reference caption per generated caption or
    multiple reference captions per generated caption. Multiple references are
    important for image captioning datasets such as Flickr and COCO.
    """
    if len(gens) != len(refs):
        raise ValueError(
            "Expected the same number of generations and references, "
            f"got {len(gens)} and {len(refs)}"
        )

    weights_map = {
        1: (1.0, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (1 / 3, 1 / 3, 1 / 3, 0),
        4: (0.25, 0.25, 0.25, 0.25),
    }
    weights = weights_map.get(n, weights_map[4])
    refs_tok = _normalize_references(refs)
    gens_tok = [tokenize(gen) for gen in gens]
    smoothing = nltk.translate.bleu_score.SmoothingFunction().method1
    try:
        return float(
            nltk.translate.bleu_score.corpus_bleu(
                refs_tok,
                gens_tok,
                weights=weights,
                smoothing_function=smoothing,
            )
        )
    except ZeroDivisionError:
        return 0.0


def compute_bleu_scores(
    gens: Sequence[str], refs: Sequence[str] | Sequence[Sequence[str]]
) -> dict[str, float]:
    """Compute BLEU-1 through BLEU-4."""
    return {f"bleu{n}": compute_bleu(gens, refs, n=n) for n in range(1, 5)}

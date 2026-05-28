"""Training entry point for CNN-LSTM image captioning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import DecoderLSTM, EncoderCNN
from utils import BOS_ID, EOS_ID, CaptionDataset, Vocabulary, compute_bleu_scores, pad_collate


def plot_curves(history: dict[str, list[float]], outpath: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (CE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_bleu(history: dict[str, list[float]], outpath: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ["val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4"]:
        ax.plot(history[key], label=key.replace("val_", "").upper())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU")
    ax.set_title("Validation BLEU Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def caption_loss(logits: torch.Tensor, targets: torch.Tensor, criterion) -> torch.Tensor:
    """Compute aligned decoder loss.

    ``DecoderLSTM.forward(features, captions[:, :-1])`` returns one extra logit
    at position 0 for the image-feature warm-up step. Ignore that logit and
    train positions 1..T to predict ``captions[:, 1:]``.
    """
    logits = logits[:, 1:, :]
    targets = targets[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def evaluate(enc, dec, dataloader, vocab, criterion, device, max_len: int, desc: str):
    enc.eval()
    dec.eval()
    total_loss = 0.0
    total_items = 0
    gens, refs = [], []

    if len(dataloader.dataset) == 0:
        return {"loss": None, "bleu1": None, "bleu2": None, "bleu3": None, "bleu4": None}

    with torch.no_grad():
        for imgs, tgt, _ in tqdm(dataloader, desc=desc):
            imgs, tgt = imgs.to(device), tgt.to(device)
            feats = enc(imgs)
            logits = dec(feats, tgt[:, :-1])
            loss = caption_loss(logits, tgt, criterion)
            total_loss += loss.item() * imgs.size(0)
            total_items += imgs.size(0)

            out_ids = dec.sample(feats, max_len=max_len, bos_id=BOS_ID, eos_id=EOS_ID)
            for i in range(out_ids.size(0)):
                gens.append(vocab.decode(out_ids[i].cpu().numpy()))
                refs.append(vocab.decode(tgt[i, 1:].cpu().numpy()))

    metrics = {"loss": total_loss / max(1, total_items)}
    metrics.update(compute_bleu_scores(gens, refs))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="CSV with columns: image_path, caption, split")
    ap.add_argument("--images-root", type=str, default="data")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-backbone", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.captions)
    required_cols = {"image_path", "caption", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"captions CSV is missing columns: {sorted(missing)}")

    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(df[df["split"] == "train"]["caption"].tolist())
    vocab.to_json(outdir / "vocab.json")

    train_ds = CaptionDataset(df, args.images_root, vocab, split="train", max_len=args.max_len, image_size=args.image_size)
    val_ds = CaptionDataset(df, args.images_root, vocab, split="val", max_len=args.max_len, image_size=args.image_size)
    test_ds = CaptionDataset(df, args.images_root, vocab, split="test", max_len=args.max_len, image_size=args.image_size)

    if len(train_ds) == 0:
        raise ValueError("No training rows found. The captions CSV must contain split='train'.")
    if len(val_ds) == 0:
        raise ValueError("No validation rows found. The captions CSV must contain split='val'.")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = EncoderCNN(
        embed_dim=args.embed_dim,
        pretrained=args.pretrained,
        train_backbone=args.train_backbone,
    ).to(device)
    dec = DecoderLSTM(
        vocab_size=len(vocab.word2id),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(list(dec.parameters()) + enc.trainable_parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_bleu1": [],
        "val_bleu2": [],
        "val_bleu3": [],
        "val_bleu4": [],
    }
    best_bleu4 = -1.0

    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        train_loss = 0.0
        train_items = 0

        for imgs, tgt, _ in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            imgs, tgt = imgs.to(device), tgt.to(device)
            optimizer.zero_grad()
            feats = enc(imgs)
            logits = dec(feats, tgt[:, :-1])
            loss = caption_loss(logits, tgt, criterion)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_items += imgs.size(0)

        train_loss /= max(1, train_items)
        val_metrics = evaluate(enc, dec, val_dl, vocab, criterion, device, args.max_len, f"Epoch {epoch}/{args.epochs} [val]")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        for n in range(1, 5):
            history[f"val_bleu{n}"].append(val_metrics[f"bleu{n}"])

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"BLEU-1={val_metrics['bleu1']:.4f} BLEU-4={val_metrics['bleu4']:.4f}"
        )

        if val_metrics["bleu4"] > best_bleu4:
            best_bleu4 = val_metrics["bleu4"]
            torch.save(
                {
                    "encoder": enc.state_dict(),
                    "decoder": dec.state_dict(),
                    "vocab_size": len(vocab.word2id),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "max_len": args.max_len,
                    "image_size": args.image_size,
                    "pretrained": args.pretrained,
                    "train_backbone": args.train_backbone,
                },
                outdir / "best_captioner.pt",
            )

        plot_curves(history, outdir / "training_curves.png")
        plot_bleu(history, outdir / "bleu_scores.png")

    metrics = {"best_val_bleu4": best_bleu4, "history": history}

    checkpoint_path = outdir / "best_captioner.pt"
    if len(test_ds) > 0 and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        enc.load_state_dict(checkpoint["encoder"])
        dec.load_state_dict(checkpoint["decoder"])
        metrics["test"] = evaluate(enc, dec, test_dl, vocab, criterion, device, args.max_len, "test")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Training done. Best validation BLEU-4: {best_bleu4:.4f}")
    if "test" in metrics:
        print(f"[OK] Test BLEU-4: {metrics['test']['bleu4']:.4f}")


if __name__ == "__main__":
    main()

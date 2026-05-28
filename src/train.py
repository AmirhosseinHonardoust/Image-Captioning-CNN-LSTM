"""Training entry point for CNN-LSTM image captioning."""

from __future__ import annotations

import argparse
import csv
import json
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


def references_by_image(dataset: CaptionDataset, vocab: Vocabulary, max_len: int) -> dict[str, list[str]]:
    """Group all reference captions for each image in an evaluation split."""
    grouped: dict[str, list[str]] = {}
    for image_path, rows in dataset.df.groupby("image_path", sort=False):
        grouped[str(image_path)] = [
            vocab.decode(vocab.encode(caption, max_len=max_len)[1:]) for caption in rows["caption"].tolist()
        ]
    return grouped


def save_predictions(predictions: list[dict[str, object]], outdir: Path) -> None:
    """Save generated captions in JSON and CSV formats for manual inspection."""
    if not predictions:
        return

    with open(outdir / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    with open(outdir / "sample_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "generated_caption", "references"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "generated_caption": row["generated_caption"],
                    "references": " | ".join(row["references"]),
                }
            )


def evaluate(
    enc,
    dec,
    dataloader,
    vocab: Vocabulary,
    criterion,
    device,
    max_len: int,
    desc: str,
    return_predictions: bool = False,
):
    enc.eval()
    dec.eval()
    total_loss = 0.0
    total_items = 0
    gens: list[str] = []
    refs: list[list[str]] = []
    predictions: list[dict[str, object]] = []
    seen_images: set[str] = set()

    if len(dataloader.dataset) == 0:
        metrics = {
            "loss": None,
            "bleu1": None,
            "bleu2": None,
            "bleu3": None,
            "bleu4": None,
            "num_samples": 0,
            "num_images": 0,
        }
        
        if return_predictions:
            metrics["predictions"] = []
        return metrics

    reference_lookup = references_by_image(dataloader.dataset, vocab, max_len)
    rows = dataloader.dataset.df.reset_index(drop=True)
    row_cursor = 0

    with torch.no_grad():
        for imgs, tgt, _ in tqdm(dataloader, desc=desc):
            batch_size = imgs.size(0)
            batch_rows = rows.iloc[row_cursor : row_cursor + batch_size]
            row_cursor += batch_size

            imgs, tgt = imgs.to(device), tgt.to(device)
            feats = enc(imgs)
            logits = dec(feats, tgt[:, :-1])
            loss = caption_loss(logits, tgt, criterion)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            out_ids = dec.sample(feats, max_len=max_len, bos_id=BOS_ID, eos_id=EOS_ID)
            for i, row in enumerate(batch_rows.itertuples(index=False)):
                image_path = str(row.image_path)
                
                # Score each image only once, even if the dataset has multiple captions
                # for the same image.
                if image_path in seen_images:
                    continue
                seen_images.add(image_path)

                generated = vocab.decode(out_ids[i].cpu().numpy())
                references = reference_lookup.get(image_path, [vocab.decode(tgt[i, 1:].cpu().numpy())])

                gens.append(generated)
                refs.append(references)

                if return_predictions:
                    predictions.append(
                        {
                            "image_path": image_path,
                            "generated_caption": generated,
                            "references": references,
                        }
                    )

    metrics = {
        "loss": total_loss / max(1, total_items),
        "num_samples": total_items,
        "num_images": len(gens),
    }
    metrics.update(compute_bleu_scores(gens, refs))
    if return_predictions:
        metrics["predictions"] = predictions
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
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm. Use 0 to disable clipping.")
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation BLEU-4 improvement. Use 0 to disable.",
    )
    ap.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum BLEU-4 improvement required to reset early stopping patience.",
    )
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
    trainable_params = list(dec.parameters()) + enc.trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_bleu1": [],
        "val_bleu2": [],
        "val_bleu3": [],
        "val_bleu4": [],
    }
    best_bleu4 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

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
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
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

        improved = val_metrics["bleu4"] > best_bleu4 + args.early_stopping_min_delta
        if improved:
            best_bleu4 = val_metrics["bleu4"]
            best_epoch = epoch
            epochs_without_improvement = 0
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
                    "best_epoch": best_epoch,
                    "best_val_bleu4": best_bleu4,
                },
                outdir / "best_captioner.pt",
            )
        else:
            epochs_without_improvement += 1

        plot_curves(history, outdir / "training_curves.png")
        plot_bleu(history, outdir / "bleu_scores.png")

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"[early stopping] No validation BLEU-4 improvement for "
                f"{args.early_stopping_patience} epoch(s). Best epoch: {best_epoch}."
            )
            break

    metrics = {"best_val_bleu4": best_bleu4, "best_epoch": best_epoch, "history": history}

    checkpoint_path = outdir / "best_captioner.pt"
    if len(test_ds) > 0 and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        enc.load_state_dict(checkpoint["encoder"])
        dec.load_state_dict(checkpoint["decoder"])
        test_metrics = evaluate(enc, dec, test_dl, vocab, criterion, device, args.max_len, "test", return_predictions=True)
        predictions = test_metrics.pop("predictions", [])
        metrics["test"] = test_metrics
        save_predictions(predictions, outdir)

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Training done. Best validation BLEU-4: {best_bleu4:.4f} at epoch {best_epoch}")
    if "test" in metrics:
        print(f"[OK] Test BLEU-4: {metrics['test']['bleu4']:.4f}")
        print(f"[OK] Sample predictions saved to {outdir / 'sample_predictions.json'}")


if __name__ == "__main__":
    main()

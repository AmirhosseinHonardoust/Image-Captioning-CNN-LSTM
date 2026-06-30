"""Inference entry point for CNN-LSTM image captioning."""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from models import DecoderLSTM, EncoderCNN
from utils import BOS, EOS, Vocabulary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Use >1 for beam search; default is greedy decoding.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vocab = Vocabulary.from_json(args.vocab)

    # The checkpoint contains the full encoder state, including the ResNet
    # backbone, so inference does not need to download pretrained weights.
    enc = EncoderCNN(
        embed_dim=checkpoint["embed_dim"],
        pretrained=False,
        train_backbone=checkpoint.get("train_backbone", False),
    ).to(device)
    dec = DecoderLSTM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)

    enc.load_state_dict(checkpoint["encoder"])
    dec.load_state_dict(checkpoint["decoder"])
    enc.eval()
    dec.eval()

    image_size = checkpoint.get("image_size", 224)
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = enc(img)
        if args.beam_size > 1:
            ids = dec.beam_search(
                feats,
                max_len=args.max_len,
                bos_id=vocab.word2id[BOS],
                eos_id=vocab.word2id[EOS],
                beam_size=args.beam_size,
            )
        else:
            ids = dec.sample(
                feats,
                max_len=args.max_len,
                bos_id=vocab.word2id[BOS],
                eos_id=vocab.word2id[EOS],
            )

    print(vocab.decode(ids[0].cpu().numpy()))


if __name__ == "__main__":
    main()

import json

import torch

from models import DecoderLSTM
from train import caption_loss
from utils import Vocabulary, compute_bleu_scores


def test_vocabulary_json_roundtrip(tmp_path):
    vocab = Vocabulary(min_freq=1)
    vocab.build(["A blue square", "A red square"])
    path = tmp_path / "vocab.json"
    vocab.to_json(path)

    loaded = Vocabulary.from_json(path)

    assert loaded.word2id == vocab.word2id
    assert loaded.id2word == vocab.id2word
    assert loaded.decode(loaded.encode("A blue square")) == "a blue square"


def test_legacy_vocab_json_loads(tmp_path):
    path = tmp_path / "legacy_vocab.json"
    path.write_text(
        json.dumps(
            {"min_freq": 1, "word2id": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "blue": 4}}
        )
    )

    loaded = Vocabulary.from_json(path)

    assert loaded.id2word[4] == "blue"


def test_decoder_alignment_shapes():
    decoder = DecoderLSTM(vocab_size=12, embed_dim=8, hidden_dim=16, dropout=0.0)
    features = torch.randn(2, 8)
    target = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    logits = decoder(features, target[:, :-1])
    aligned_logits = logits[:, 1:, :]

    assert aligned_logits.shape == (2, target[:, 1:].shape[1], 12)


def test_caption_loss_ignores_image_feature_logit():
    decoder = DecoderLSTM(vocab_size=12, embed_dim=8, hidden_dim=16, dropout=0.0)
    features = torch.randn(2, 8)
    target = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    logits = decoder(features, target[:, :-1])
    loss = caption_loss(logits, target, torch.nn.CrossEntropyLoss(ignore_index=0))

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bleu_scores_has_all_keys():
    scores = compute_bleu_scores(["a blue square"], ["a blue square"])

    assert set(scores) == {"bleu1", "bleu2", "bleu3", "bleu4"}
    assert scores["bleu1"] > 0


def test_bleu_scores_support_multiple_references():
    scores = compute_bleu_scores(["a blue square"], [["a blue square", "a small blue shape"]])

    assert set(scores) == {"bleu1", "bleu2", "bleu3", "bleu4"}
    assert scores["bleu1"] > 0


def test_pad_collate_returns_indices_aligned_with_samples():
    from utils import pad_collate

    batch = [
        (torch.zeros(3, 4, 4), torch.tensor([1, 4, 2]), 5),
        (torch.zeros(3, 4, 4), torch.tensor([1, 6, 7, 2]), 9),
    ]
    imgs, padded, lengths, indices = pad_collate(batch)

    assert imgs.shape == (2, 3, 4, 4)
    assert padded.shape == (2, 4)  # padded to the longest sequence
    assert lengths.tolist() == [3, 4]
    assert indices.tolist() == [5, 9]  # original dataset row indices preserved


class _TinyEncoder(torch.nn.Module):
    """Deterministic stand-in for the CNN encoder (avoids a ResNet download)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def _build_eval_dataset(tmp_path):
    import pandas as pd
    from PIL import Image

    from utils import CaptionDataset, Vocabulary

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    colors = {
        "a.png": (255, 0, 0),
        "b.png": (0, 255, 0),
        "c.png": (0, 0, 255),
        "d.png": (255, 255, 0),
    }
    for name, color in colors.items():
        Image.new("RGB", (16, 16), color).save(img_dir / name)

    df = pd.DataFrame(
        {
            "image_path": [
                "images/a.png",
                "images/a.png",
                "images/b.png",
                "images/c.png",
                "images/c.png",
                "images/d.png",
            ],
            "caption": [
                "a red square",
                "a crimson block",
                "a green square",
                "a blue square",
                "a navy tile",
                "a yellow square",
            ],
            "split": ["test"] * 6,
        }
    )
    vocab = Vocabulary(min_freq=1)
    vocab.build(df["caption"].tolist())
    ds = CaptionDataset(df, images_root=str(tmp_path), vocab=vocab, split="test", image_size=16)
    return ds, vocab


def _run_eval(ds, vocab, shuffle, seed):
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from train import evaluate
    from utils import pad_collate

    torch.manual_seed(0)
    enc = _TinyEncoder(embed_dim=8)
    enc.eval()
    dec = DecoderLSTM(vocab_size=len(vocab.word2id), embed_dim=8, hidden_dim=16, dropout=0.0)
    dec.eval()

    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=shuffle,
        collate_fn=pad_collate,
        num_workers=0,
        generator=gen if shuffle else None,
    )
    metrics = evaluate(
        enc,
        dec,
        dl,
        vocab,
        nn.CrossEntropyLoss(ignore_index=0),
        torch.device("cpu"),
        max_len=12,
        desc="test",
        return_predictions=True,
    )
    return sorted(metrics["predictions"], key=lambda p: p["image_path"])


def test_evaluate_predictions_are_order_invariant(tmp_path):
    """evaluate() must map predictions to images by identity, not loader order."""
    ds, vocab = _build_eval_dataset(tmp_path)
    ordered = _run_eval(ds, vocab, shuffle=False, seed=0)
    shuffled_a = _run_eval(ds, vocab, shuffle=True, seed=7)
    shuffled_b = _run_eval(ds, vocab, shuffle=True, seed=123)

    assert ordered == shuffled_a == shuffled_b
    # Each unique image is scored exactly once.
    assert len(ordered) == 4
    for pred in ordered:
        assert pred["generated_caption"]  # non-empty decode


def test_beam_search_is_deterministic_and_shaped():
    torch.manual_seed(0)
    dec = DecoderLSTM(vocab_size=12, embed_dim=8, hidden_dim=16, dropout=0.0)
    dec.eval()
    feats = torch.randn(1, 8)

    with torch.no_grad():
        first = dec.beam_search(feats, max_len=6, bos_id=1, eos_id=2, beam_size=3)
        second = dec.beam_search(feats, max_len=6, bos_id=1, eos_id=2, beam_size=3)

    assert first.shape == (1, 6)
    assert torch.equal(first, second)


def test_attention_forward_shapes_and_alpha_normalized():
    from models import AttentionDecoderLSTM

    torch.manual_seed(0)
    dec = AttentionDecoderLSTM(
        vocab_size=12, embed_dim=16, hidden_dim=32, encoder_dim=24, attention_dim=16, dropout=0.0
    )
    encoder_out = torch.randn(2, 5, 24)  # (batch, num_pixels, encoder_dim)
    captions = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])

    logits, alphas = dec(encoder_out, captions)

    assert logits.shape == (2, 4, 12)
    assert alphas.shape == (2, 4, 5)  # attention over the 5 spatial locations
    # Each timestep's attention is a distribution over spatial locations.
    assert torch.allclose(alphas.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_attention_decoder_sample_shape():
    from models import AttentionDecoderLSTM

    torch.manual_seed(0)
    dec = AttentionDecoderLSTM(
        vocab_size=12, embed_dim=16, hidden_dim=32, encoder_dim=24, attention_dim=16, dropout=0.0
    )
    dec.eval()
    encoder_out = torch.randn(3, 5, 24)

    ids = dec.sample(encoder_out, max_len=6, bos_id=1, eos_id=2)

    assert ids.shape[0] == 3
    assert ids.shape[1] <= 6


def test_attention_decoder_learns_tiny_batch():
    """The attention decoder should overfit a tiny fixed mapping (sanity that it learns)."""
    import torch.nn as nn

    from models import AttentionDecoderLSTM

    torch.manual_seed(0)
    vocab, bos, eos = 12, 1, 2
    encoder_out = torch.randn(3, 4, 24)
    captions = torch.tensor([[bos, 3, 4, 5, eos], [bos, 6, 7, eos, 0], [bos, 8, 9, 3, eos]])
    dec = AttentionDecoderLSTM(
        vocab_size=vocab, embed_dim=16, hidden_dim=32, encoder_dim=24, attention_dim=16, dropout=0.0
    )
    opt = torch.optim.Adam(dec.parameters(), lr=0.05)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    inp, tgt = captions[:, :-1], captions[:, 1:]

    initial = None
    loss = torch.tensor(0.0)
    for _ in range(200):
        opt.zero_grad()
        logits, alphas = dec(encoder_out, inp)
        loss = crit(logits.reshape(-1, vocab), tgt.reshape(-1))
        loss = loss + ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()

    assert loss.item() < 0.5  # converged well below the starting cross-entropy
    assert loss.item() < initial


def test_encoder_attention_output_shape():
    """EncoderCNNAttention returns a flattened spatial grid (B, num_pixels, encoder_dim)."""
    from models import EncoderCNNAttention

    enc = EncoderCNNAttention(pretrained=False, train_backbone=False, encoded_image_size=2)
    enc.eval()
    imgs = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        feats = enc(imgs)

    assert feats.shape == (2, 4, enc.encoder_dim)  # 2x2 grid -> 4 pixels
    assert enc.encoder_dim == 2048

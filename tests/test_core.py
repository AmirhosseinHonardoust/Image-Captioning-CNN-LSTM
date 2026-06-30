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

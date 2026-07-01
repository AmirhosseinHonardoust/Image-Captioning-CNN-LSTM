<div align="center">

# Image Captioning with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c) ![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-purple) ![NLP](https://img.shields.io/badge/NLP-LSTM-green) ![Task](https://img.shields.io/badge/Task-Image%20Captioning-orange) ![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-brightgreen) ![License](https://img.shields.io/github/license/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM) [![CI](https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM/actions/workflows/ci.yml)

</div>

A deep learning project that generates natural-language captions for images using a **CNN encoder** and an **LSTM decoder**, with an optional **visual attention** decoder.

The project combines computer vision and natural language processing by using a **ResNet-50 image encoder** to extract visual features and an **LSTM language decoder** to generate captions word by word.

> **Important:** This project is an educational CNN-LSTM image captioning baseline.
>
> The included demo dataset is intentionally tiny and is designed to verify the training and inference pipeline, not to represent real-world captioning performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [How the Model Works](#how-the-model-works)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Model Output](#model-output)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Reports](#visual-reports)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Image captioning is a multimodal deep learning task that connects **computer vision** and **language generation**.

Given an input image, the model learns to generate a text caption that describes the visual content of the image.

This project uses a classic encoder-decoder architecture, with an optional attention variant:

```text
Image
        ↓
CNN Encoder
        ↓
Image Feature(s)
        ↓
LSTM Decoder (optional attention)
        ↓
Generated Caption
```

The goal of this project is to demonstrate:

- A clean deep learning workflow
- CNN-LSTM model architecture with an optional attention decoder
- Image preprocessing and caption tokenization
- Vocabulary building
- Training, validation, and test evaluation
- Greedy decoding and beam search inference
- BLEU score evaluation
- Professional ML project documentation
- Reproducible testing and experimentation

---

## What This Project Does

This project can:

- Load image-caption pairs from a CSV file
- Build a vocabulary from training captions
- Extract visual features from images using a CNN encoder
- Train an LSTM decoder to generate captions
- Train an optional attention decoder (Show, Attend and Tell)
- Evaluate generated captions using BLEU scores
- Save trained model checkpoints
- Save vocabulary files
- Generate training and validation loss curves
- Generate BLEU score plots
- Run inference on new images
- Support greedy decoding
- Support beam search decoding
- Run automated tests for core functionality

---

## What This Project Does Not Do

This project does **not**:

- Understand images like a human
- Guarantee accurate captions for real-world images without enough training data
- Produce strong results from the tiny demo dataset
- Replace modern transformer-based vision-language models
- Perform object detection or segmentation
- Use external knowledge about the image
- Verify whether generated captions are factually complete
- Provide production-level image captioning out of the box

A stronger real-world image captioning system would require a larger dataset, stronger evaluation, more diverse captions, and usually a modern attention-based or transformer-based architecture.

---

## Key Features

- **CNN-LSTM encoder-decoder architecture**
- **ResNet-50 image encoder**
- **LSTM caption decoder**
- **Optional visual attention decoder (Show, Attend and Tell)**
- **Custom vocabulary builder**
- **Caption tokenization**
- **Train / validation / test split support**
- **Cross-entropy loss training with doubly-stochastic attention regularization**
- **BLEU-1, BLEU-2, BLEU-3, and BLEU-4 evaluation**
- **Grouped multi-reference BLEU evaluation**
- **Greedy decoding and beam search**
- **Training and validation loss visualization**
- **BLEU score visualization**
- **Saved model checkpoints, vocabulary, and metrics**
- **Saved sample prediction exports**
- **Early stopping and gradient clipping options**
- **Pytest test suite**
- **Clean project structure**
- **GitHub Actions CI with ruff, black, and mypy**

---

## System Workflow

```text
Image-caption dataset (CSV)
        ↓
Vocabulary building and tokenization
        ↓
Train / validation / test split
        ↓
CNN encoder + LSTM or attention decoder
        ↓
Cross-entropy training with early stopping
        ↓
Validation and test BLEU evaluation
        ↓
Checkpoints, metrics, predictions, and charts
        ↓
Inference on new images
```

---

## How the Model Works

The project follows a classic image captioning pipeline:

```text
Input Image
        ↓
Image Preprocessing
        ↓
CNN Encoder
        ↓
Image Feature(s)
        ↓
LSTM Decoder (optional attention)
        ↓
Vocabulary Distribution
        ↓
Generated Caption
```

---

### CNN Encoder

The encoder uses a ResNet-50 backbone to extract visual features from an image.

The final classification layer is removed, and the extracted features are projected into the decoder embedding space.

```text
Image → ResNet-50 → Feature Vector → Linear Projection → Image Embedding
```

The encoder is responsible for converting the image into a compact numerical representation.

---

### LSTM Decoder

The decoder is an LSTM-based language model.

It receives the image representation and generates the caption one token at a time.

```text
Image Embedding + Previous Words → LSTM → Next Word Prediction
```

During training, the decoder learns to predict the next word in the caption sequence.

During inference, the decoder generates words until it reaches an end token or the maximum caption length.

---

### Attention Decoder (Show, Attend and Tell)

The project also includes an optional attention decoder, selected with `--decoder attention`.

Instead of compressing the image into a single vector, the encoder keeps the spatial feature grid, and the decoder attends over image regions at each step.

```text
Spatial Image Features + Previous Words → Attention → Context → LSTM → Next Word
```

At every step the decoder produces a distribution over image locations (the attention weights) and uses the resulting context vector to predict the next word. This raises the model's ceiling and makes generation interpretable, at the cost of a heavier decoder.

The baseline `lstm` decoder remains the default, and the two share the same training, evaluation, and inference code paths.

---

### Decoding Strategies

The project supports two decoding strategies.

#### Greedy Decoding

Greedy decoding selects the most likely word at each step.

```text
Choose best word → choose next best word → continue
```

It is fast and simple, but it may miss better full-sentence captions.

#### Beam Search

Beam search keeps multiple candidate captions during generation.

```text
Keep top-k caption candidates → expand candidates → choose best sequence
```

Beam search is slower than greedy decoding but can produce better captions. Beam search currently applies to the baseline decoder; the attention decoder uses greedy decoding.

---

## Dataset Format

The training script expects a CSV file with the following columns:

```csv
image_path,caption,split
images/example1.jpg,a blue square with the word example1,train
images/example1.jpg,a blue image used for training,train
images/example2.jpg,an orange square with the word example2,val
images/example2.jpg,an orange image used for testing,test
```

Column descriptions:

<div align="center">

| Column | Description |
|---|---|
| `image_path` | Relative path to the image from `--images-root` |
| `caption` | Text caption for the image |
| `split` | Dataset split: `train`, `val`, or `test` |

</div>

Multiple rows can share the same `image_path`. During evaluation, the code groups those rows as multiple reference captions for the same image, which is the standard setup for many captioning datasets.

Example folder structure:

```text
data/
├── captions.csv
└── images/
    ├── example1.jpg
    └── example2.jpg
```

If `image_path` is `images/example1.jpg` and `--images-root` is `data`, then the full image path becomes `data/images/example1.jpg`.

---

## Project Structure

```text
Image-Captioning-CNN-LSTM/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── captions.csv
│   └── images/
│       ├── example1.jpg
│       └── example2.jpg
│
├── outputs/
│   ├── best_captioner.pt          # generated after training; not committed
│   ├── vocab.json
│   ├── metrics.json
│   ├── sample_predictions.csv     # generated after test evaluation
│   ├── sample_predictions.json    # generated after test evaluation
│   ├── training_curves.png
│   └── bleu_scores.png
│
├── scripts/
│   └── prepare_flickr_csv.py
│
├── src/
│   ├── infer.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_core.py
│
├── .gitignore
├── MODEL_CARD.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM.git
cd Image-Captioning-CNN-LSTM
```

### 2. Create a Virtual Environment

Use Python 3.10 for the most reliable compatibility with the pinned PyTorch and Torchvision versions.

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development and testing tools:

```bash
pip install -r requirements-dev.txt
```

---

## Quick Start

Train on the included demo dataset and run inference in three commands:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --min-freq 1 --num-workers 0 --grad-clip 1.0 --early-stopping-patience 5
python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image data/images/example1.jpg --max-len 20
```

To try the attention decoder instead of the baseline, add `--decoder attention` to the training command.

> The repository does not ship a trained checkpoint. Run training first to generate `outputs/best_captioner.pt`, then use that checkpoint for inference.

---

## Training the Model

Run:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --min-freq 1 --num-workers 0 --grad-clip 1.0 --early-stopping-patience 5
```

This will:

- Load the captions CSV
- Build the vocabulary
- Load and preprocess images
- Train the CNN-LSTM model
- Evaluate on the validation split
- Evaluate on the test split if available
- Save the best checkpoint, the vocabulary, and metrics
- Save test-set sample predictions when a test split exists
- Generate charts

Generated outputs:

```text
outputs/best_captioner.pt
outputs/vocab.json
outputs/metrics.json
outputs/sample_predictions.csv
outputs/sample_predictions.json
outputs/training_curves.png
outputs/bleu_scores.png
```

### Common Training Arguments

<div align="center">

| Argument | Description |
|---|---|
| `--captions` | Path to the captions CSV file |
| `--images-root` | Root directory for image paths |
| `--outdir` | Directory for saved outputs |
| `--epochs` | Number of training epochs |
| `--batch-size` | Training batch size |
| `--embed-dim` | Embedding dimension |
| `--hidden-dim` | LSTM hidden dimension |
| `--num-layers` | Number of LSTM layers |
| `--dropout` | Dropout value |
| `--min-freq` | Minimum word frequency for vocabulary |
| `--max-len` | Maximum caption length |
| `--lr` | Learning rate |
| `--num-workers` | DataLoader worker count |
| `--no-pretrained` | Disable pretrained ResNet weights |
| `--train-backbone` | Fine-tune the CNN backbone |
| `--decoder` | Decoder architecture: `lstm` (default) or `attention` |
| `--attention-dim` | Attention hidden size (attention decoder only) |
| `--alpha-c` | Doubly-stochastic attention regularization weight |
| `--grad-clip` | Max gradient norm; use `0` to disable |
| `--early-stopping-patience` | Stop after N epochs without BLEU-4 improvement |
| `--early-stopping-min-delta` | Minimum BLEU-4 improvement for early stopping |

</div>

### Training on a Custom Dataset

To train on your own dataset, prepare this structure:

```text
my_dataset/
├── captions.csv
└── images/
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```

Train with:

```bash
python src/train.py --captions my_dataset/captions.csv --images-root my_dataset --epochs 20 --min-freq 1 --num-workers 0 --early-stopping-patience 5
```

For Flickr-style caption files, you can generate a compatible CSV:

```bash
python scripts/prepare_flickr_csv.py --captions-file Flickr8k.token.txt --images-subdir images --output my_dataset/captions.csv
```

To train the attention decoder on real data (for example Flickr8k), add `--decoder attention` and let the backbone fine-tune:

```bash
python src/train.py --captions my_dataset/captions.csv --images-root my_dataset --decoder attention --train-backbone --epochs 30 --min-freq 3 --early-stopping-patience 5
```

The decoder type is stored in the checkpoint, so inference automatically rebuilds the correct architecture.

---

## Running Inference

After training has created `outputs/best_captioner.pt`, run inference on a single image:

```bash
python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image data/images/example1.jpg --max-len 20
```

Using beam search:

```bash
python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image data/images/example1.jpg --max-len 20 --beam-size 3
```

Example output:

```text
a blue image used for training
```

---

## Model Output

The model returns a generated text caption.

```text
Input:  example1.jpg
Output: a blue image used for training
```

```text
Input:  example2.jpg
Output: an orange image used for training
```

The generated caption depends heavily on the training dataset. With the tiny demo dataset, the model learns only a very small vocabulary and simple captions.

---

## Evaluation Metrics

The project evaluates generated captions with corpus BLEU over grouped multi-reference captions, alongside training and validation loss.

<div align="center">

| Metric | Meaning |
|---|---|
| BLEU-1 | Unigram / single-word overlap |
| BLEU-2 | Two-word sequence overlap |
| BLEU-3 | Three-word sequence overlap |
| BLEU-4 | Four-word sequence overlap |
| Train / validation loss | Optimization and overfitting behavior |
| Best validation epoch | Epoch with the highest validation BLEU-4 |

</div>

If an evaluation split contains multiple rows for the same image, those captions are grouped as multiple references for BLEU scoring.

Example results from the included demo run:

<div align="center">

| Metric | Demo value |
|---|---|
| Best validation BLEU-4 | 0.027 |
| Best epoch | 4 |
| Test BLEU-4 (2 images) | 0.846 |
| Final train loss | 0.677 |
| Final validation loss | 3.377 |

</div>

> These values come from the tiny two-image demo dataset and reflect memorization, not real-world captioning performance. The rising validation loss is expected overfitting on so few samples.

Metrics and predictions are saved to:

```text
outputs/metrics.json
outputs/sample_predictions.csv
outputs/sample_predictions.json
```

Example training log:

```text
[epoch 4] train_loss=2.0221 val_loss=2.8193 BLEU-1=0.2500 BLEU-4=0.0269
[OK] Training done. Best validation BLEU-4: 0.0269
[OK] Test BLEU-4: 0.8465
```

---

## Visual Reports

The project automatically generates visual outputs during training, saved under `outputs/`.

<div align="center">

| Training and Validation Loss | BLEU Scores |
|---|---|
| <img width="420" alt="training_curves" src="https://github.com/user-attachments/assets/f06a3be7-93d9-4c36-a221-d40066eba77f" /> | <img width="420" alt="bleu_scores" src="https://github.com/user-attachments/assets/5673d798-3223-4c1f-8e9c-55da08a79350" /> |
| **Analysis:** Training loss decreases steadily while validation loss stays flat and then rises, and the gap widens across epochs. This is textbook overfitting on the tiny demo dataset, where the model memorizes a few samples faster than it learns general captioning patterns. | **Analysis:** BLEU-1 improves early and then plateaus, showing some word-level learning, while BLEU-2 to BLEU-4 stay low because matching longer sequences is harder. Scores fluctuate because the validation set is extremely small. |

</div>

<details>
<summary>Chart interpretation</summary>

The charts confirm that:

- The training pipeline works end-to-end
- The model can learn from image-caption pairs
- The model quickly overfits on the tiny demo dataset
- Validation metrics are unstable when the validation set is too small
- A larger dataset is required for meaningful performance evaluation

A healthier curve on a larger dataset would usually show both training and validation loss decreasing for several epochs before validation loss stabilizes.

</details>

---

## Testing and CI

Run the test suite:

```bash
pytest
```

Expected result:

```text
13 passed
```

The tests check important project behavior, including:

- Vocabulary serialization and loading
- BLEU score output and multi-reference support
- Decoder output alignment and loss masking
- Order-independent evaluation (predictions map to images by identity)
- Attention decoder shapes, attention normalization, and learning behavior
- Attention encoder output shape

Run the full quality gate locally (matches CI):

```bash
pytest -q
ruff check .
black --check .
mypy
```

The GitHub Actions workflow runs tests, linting (ruff), formatting (black), and type checking (mypy) on pushes and pull requests. CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project keeps a clean, maintainable structure:

<div align="center">

| File | Purpose |
|---|---|
| `src/models.py` | CNN encoder, LSTM decoder, attention encoder/decoder, beam search |
| `src/utils.py` | Vocabulary, tokenization, dataset, collation, and BLEU helpers |
| `src/train.py` | Training loop, evaluation, and metric/checkpoint export |
| `src/infer.py` | Single-image inference with greedy or beam decoding |
| `scripts/prepare_flickr_csv.py` | Convert Flickr-style caption files to the project CSV |
| `tests/test_core.py` | Unit tests for core behavior |

</div>

Tooling is configured in `pyproject.toml` and `requirements-dev.txt`, with ruff, black, and mypy enforced in CI. Recommended checks before pushing:

```bash
pytest
ruff check .
black --check .
mypy
```

---

## Limitations

This project has important limitations.

The model:

- Is a CNN-LSTM baseline
- Is weaker than modern transformer-based image captioning models
- Needs a larger dataset for meaningful captions
- Overfits quickly on tiny datasets
- Has unstable BLEU scores when validation data is too small
- Includes an attention decoder, but does not include a trained large-scale checkpoint
- Does not use object detection
- Does not reason about unseen objects
- Does not verify caption correctness
- Should not be treated as a production-ready captioning system

High performance on a tiny demo dataset does not guarantee useful real-world captioning performance.

---

## Responsible Use

This project is intended for:

- Deep learning education
- Computer vision practice
- NLP practice
- Multimodal AI portfolio demonstration
- CNN-LSTM and attention architecture learning
- Reproducible ML experimentation

It should not be used for:

- Production image understanding
- Accessibility tools without stronger validation
- Safety-critical visual interpretation
- Medical, legal, or security image analysis
- Replacing human review

Generated captions can be incomplete, biased, or incorrect depending on the training data.

---

## Future Improvements

Possible future improvements include:

- Train on a larger custom image-caption dataset
- Add CIDEr, METEOR, and ROUGE-L evaluation
- Add a transformer-based decoder
- Add pretrained vision-language models
- Add attention-map visualization for generated captions
- Add experiment tracking
- Add richer dataset analysis scripts
- Add a data statement
- Add a Streamlit or Gradio demo app
- Add Docker support
- Add a notebook walkthrough

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- pandas
- NumPy
- Pillow
- NLTK
- Matplotlib
- tqdm
- pytest

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License.

This project is intended for educational and portfolio purposes. If you use or modify it, keep the dataset limitations and model limitations clear.

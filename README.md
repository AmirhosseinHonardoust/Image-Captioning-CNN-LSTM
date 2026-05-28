<div align="center">
    
# Image Captioning with CNN-LSTM
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c) ![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-purple) ![NLP](https://img.shields.io/badge/NLP-LSTM-green) ![Task](https://img.shields.io/badge/Task-Image%20Captioning-orange) ![Status](https://img.shields.io/badge/Status-Educational%20ML%20Project-brightgreen) ![License](https://img.shields.io/github/license/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM) [![CI](https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM/actions/workflows/ci.yml)

</div>

A professional deep learning project that generates natural-language captions for images using a **CNN encoder** and an **LSTM decoder**.

The project combines computer vision and natural language processing by using a **ResNet-50 image encoder** to extract visual features and an **LSTM language decoder** to generate captions word by word.

> **Important:** This project is an educational CNN-LSTM image captioning baseline.
> The included demo dataset is intentionally tiny and is designed to verify the training and inference pipeline, not to represent real-world captioning performance.

---

## Table of Contents

* [Project Overview](#project-overview)
* [What This Project Does](#what-this-project-does)
* [What This Project Does Not Do](#what-this-project-does-not-do)
* [Features](#features)
* [Demo Result](#demo-result)
* [Charts and Visual Analysis](#charts-and-visual-analysis)
* [How the Model Works](#how-the-model-works)
* [Dataset Format](#dataset-format)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Training the Model](#training-the-model)
* [Running Inference](#running-inference)
* [Model Output](#model-output)
* [Evaluation](#evaluation)
* [Testing](#testing)
* [Code Quality](#code-quality)
* [Limitations](#limitations)
* [Responsible Use](#responsible-use)
* [Future Improvements](#future-improvements)
* [Tech Stack](#tech-stack)
* [Author](#author)
* [License](#license)

---

## Project Overview

Image captioning is a multimodal deep learning task that connects **computer vision** and **language generation**.

Given an input image, the model learns to generate a text caption that describes the visual content of the image.

This project uses a classic encoder-decoder architecture:

```text
Image
↓
CNN Encoder
↓
Image Feature Vector
↓
LSTM Decoder
↓
Generated Caption
```

The goal of this project is to demonstrate:

* A clean deep learning workflow
* CNN-LSTM model architecture
* Image preprocessing and caption tokenization
* Vocabulary building
* Training, validation, and test evaluation
* Greedy decoding and beam search inference
* BLEU score evaluation
* Professional ML project documentation
* Reproducible testing and experimentation

---

## What This Project Does

This project can:

* Load image-caption pairs from a CSV file
* Build a vocabulary from training captions
* Extract visual features from images using a CNN encoder
* Train an LSTM decoder to generate captions
* Evaluate generated captions using BLEU scores
* Save trained model checkpoints
* Save vocabulary files
* Generate training and validation loss curves
* Generate BLEU score plots
* Run inference on new images
* Support greedy decoding
* Support beam search decoding
* Run automated tests for core functionality

---

## What This Project Does Not Do

This project does **not**:

* Understand images like a human
* Guarantee accurate captions for real-world images without enough training data
* Produce strong results from the tiny demo dataset
* Replace modern transformer-based vision-language models
* Perform object detection or segmentation
* Use external knowledge about the image
* Verify whether generated captions are factually complete
* Provide production-level image captioning out of the box

A stronger real-world image captioning system would require a larger dataset, stronger evaluation, more diverse captions, and usually a modern attention-based or transformer-based architecture.

---

## Features

* **CNN-LSTM encoder-decoder architecture**
* **ResNet-50 image encoder**
* **LSTM caption decoder**
* **Custom vocabulary builder**
* **Caption tokenization**
* **Train / validation / test split support**
* **Cross-entropy loss training**
* **BLEU-1, BLEU-2, BLEU-3, and BLEU-4 evaluation**
* **Greedy decoding**
* **Beam search decoding**
* **Training and validation loss visualization**
* **BLEU score visualization**
* **Saved model checkpoints**
* **Saved vocabulary file**
* **Saved metrics file**
* **Pytest test suite**
* **Clean project structure**
* **Professional README documentation**
* **GitHub Actions CI and Ruff configuration**
* **Grouped multi-reference BLEU evaluation**
* **Early stopping and gradient clipping options**
* **Saved sample prediction exports**

---

## Demo Result

The included demo dataset is intentionally small. It is used to confirm that the full pipeline works:

```text
dataset → training → checkpoint → inference → generated caption
```

Example inference results after training on the tiny demo dataset:

```text
example1.jpg → a blue image used for training
example2.jpg → an orange image used for training
```

These results show that the model learned to distinguish between the two demo images.

However, this is not real-world captioning performance. With only a few images, the model mainly memorizes the training examples.

---

## Charts and Visual Analysis

The project automatically generates visual outputs during training. These charts help explain how the model behaves over time.

Generated charts are saved in:

```text
outputs/
```

Main visual outputs:

| Chart                        | Path                          | Purpose                                     |
| ---------------------------- | ----------------------------- | ------------------------------------------- |
| Training and Validation Loss | `outputs/training_curves.png` | Shows optimization behavior and overfitting |
| BLEU Scores                  | `outputs/bleu_scores.png`     | Shows caption quality metrics across epochs |

---

### Training and Validation Loss

<img width="650" height="400" alt="training_curves" src="https://github.com/user-attachments/assets/f06a3be7-93d9-4c36-a221-d40066eba77f" />

The loss curve shows how the model performs during training and validation.

In the demo run:

Training loss decreases steadily
Validation loss stays almost flat early, then increases over time
The gap between training and validation loss becomes larger across epochs

This indicates overfitting.

The model is learning the tiny training dataset faster than it is learning general image-captioning patterns. This behavior is expected because the demo dataset contains only a few samples.

A healthier curve on a larger dataset would usually show both training and validation loss decreasing for several epochs before validation loss stabilizes.

---

### BLEU Score Evaluation

<img width="650" height="400" alt="bleu_scores" src="https://github.com/user-attachments/assets/5673d798-3223-4c1f-8e9c-55da08a79350" />

The BLEU chart shows validation BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores across epochs.

BLEU scores measure overlap between generated captions and reference captions.

| Metric | Meaning                                |
| ------ | -------------------------------------- |
| BLEU-1 | Measures unigram / single-word overlap |
| BLEU-2 | Measures two-word sequence overlap     |
| BLEU-3 | Measures three-word sequence overlap   |
| BLEU-4 | Measures four-word sequence overlap    |

BLEU-1 improves during the first few epochs and then plateaus, showing that the model learns some individual word-level patterns.

BLEU-2, BLEU-3, and BLEU-4 remain much lower because matching longer word sequences is harder, especially with a very small validation set.

In the demo run, BLEU scores fluctuate because the validation set is extremely small. With only a few validation examples, a small change in the generated caption can cause a large change in BLEU.


---

### Chart Interpretation

The charts confirm that:

* The training pipeline works end-to-end
* The model can learn from image-caption pairs
* The model quickly overfits on the tiny demo dataset
* Validation metrics are unstable when the validation set is too small
* A larger dataset is required for meaningful performance evaluation

This is the expected result for a small educational demo.

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
Image Feature Vector
↓
LSTM Decoder
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

Beam search is slower than greedy decoding but can produce better captions.

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

| Column       | Description                                     |
| ------------ | ----------------------------------------------- |
| `image_path` | Relative path to the image from `--images-root` |
| `caption`    | Text caption for the image                      |
| `split`      | Dataset split: `train`, `val`, or `test`        |

Example folder structure:

```text
data/
├── captions.csv
└── images/
    ├── example1.jpg
    └── example2.jpg
```

If `image_path` is:

```text
images/example1.jpg
```

and `--images-root` is:

```text
data
```

then the full image path becomes:

```text
data/images/example1.jpg
```

---

## Project Structure

```text
Image-Captioning-CNN-LSTM/
│
├── data/
│   ├── captions.csv
│   └── images/
│       ├── example1.jpg
│       └── example2.jpg
│
├── .github/
│   └── workflows/
│       └── ci.yml
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

---

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

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development and testing tools:

```bash
pip install -r requirements-dev.txt
```

---

## Training the Model

Run:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --min-freq 1 --num-workers 0 --grad-clip 1.0 --early-stopping-patience 5
```

> The repository does not ship a trained checkpoint by default. Run training first to generate `outputs/best_captioner.pt`, then use that checkpoint for inference.

This will:

* Load the captions CSV
* Build the vocabulary
* Load and preprocess images
* Train the CNN-LSTM model
* Evaluate on the validation split
* Evaluate on the test split if available
* Save the best checkpoint
* Save the vocabulary
* Save metrics
* Save test-set sample predictions when a test split exists
* Generate charts

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

---

### Common Training Arguments

| Argument | Description |
| --- | --- |
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
| `--grad-clip` | Max gradient norm; use `0` to disable |
| `--early-stopping-patience` | Stop after N epochs without BLEU-4 improvement |
| `--early-stopping-min-delta` | Minimum BLEU-4 improvement for early stopping |

---

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

Example CSV:

```csv
image_path,caption,split
images/img1.jpg,a cat sitting on a chair,train
images/img1.jpg,a small cat resting indoors,train
images/img2.jpg,a dog running outside,val
images/img2.jpg,a brown dog playing outdoors,val
images/img3.jpg,a red car parked on the road,test
```

Multiple rows can share the same `image_path`. During evaluation, the code groups those rows as multiple reference captions for the same image, which is the standard setup for many captioning datasets.

Train with:

```bash
python src/train.py --captions my_dataset/captions.csv --images-root my_dataset --epochs 20 --min-freq 1 --num-workers 0 --early-stopping-patience 5
```

For Flickr-style caption files, you can generate a compatible CSV:

```bash
python scripts/prepare_flickr_csv.py --captions-file Flickr8k.token.txt --images-subdir images --output my_dataset/captions.csv
```

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

Example:

```text
Input:  example1.jpg
Output: a blue image used for training
```

Another example:

```text
Input:  example2.jpg
Output: an orange image used for training
```

The generated caption depends heavily on the training dataset. With the tiny demo dataset, the model learns only a very small vocabulary and simple captions.

---

## Evaluation

The project uses an evaluation workflow based on caption overlap metrics.

Evaluation includes:

* Training loss
* Validation loss
* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* Best validation epoch
* Test BLEU-4 when a test split is available
* Sample generated captions for manual review

If an evaluation split contains multiple rows for the same image, those captions are grouped as multiple references for BLEU scoring.

Metrics are saved to:

```text
outputs/metrics.json
outputs/sample_predictions.csv
outputs/sample_predictions.json
```

Charts are saved to:

```text
outputs/training_curves.png
outputs/bleu_scores.png
```

Example training log:

```text
[epoch 1] train_loss=1.2785 val_loss=0.7448 BLEU-1=0.3250 BLEU-4=0.2638
[OK] Training done. Best validation BLEU-4: 0.2638
[OK] Test BLEU-4: 0.1792
```

---

### Why This Matters

A common mistake in beginner image captioning projects is showing generated captions without explaining how reliable the evaluation is.

This project reports training behavior, validation behavior, BLEU metrics, and limitations so the results are easier to interpret.

For tiny datasets, the metrics should not be treated as real performance indicators.

---

## Testing

Run the test suite:

```bash
pytest
```

Expected result:

```text
6 passed
```

The tests check important project behavior, including:

* Vocabulary serialization and loading
* BLEU score output
* Decoder output alignment
* Core model behavior

---

## Code Quality

The project includes development tooling through:

```text
requirements-dev.txt
pyproject.toml
tests/
.github/workflows/ci.yml
```

These files support:

* Automated tests
* More reliable refactoring
* Cleaner project maintenance
* Professional GitHub presentation

Recommended checks before pushing changes:

```bash
pytest
ruff check .
```

The repository also includes GitHub Actions CI in `.github/workflows/ci.yml` so tests and linting can run automatically on pushes and pull requests.

---

## Limitations

This project has important limitations.

The model:

* Is a CNN-LSTM baseline
* Is weaker than modern transformer-based image captioning models
* Needs a larger dataset for meaningful captions
* Overfits quickly on tiny datasets
* Has unstable BLEU scores when validation data is too small
* Does not use attention
* Does not use object detection
* Does not reason about unseen objects
* Does not verify caption correctness
* Should not be treated as a production-ready captioning system

High performance on a tiny demo dataset does not guarantee useful real-world captioning performance.

---

## Responsible Use

This project is intended for:

* Deep learning education
* Computer vision practice
* NLP practice
* Multimodal AI portfolio demonstration
* CNN-LSTM architecture learning
* Reproducible ML experimentation

It should not be used for:

* Production image understanding
* Accessibility tools without stronger validation
* Safety-critical visual interpretation
* Medical, legal, or security image analysis
* Replacing human review

Generated captions can be incomplete, biased, or incorrect depending on the training data.

---

## Future Improvements

Possible future improvements include:

* Train on a larger custom image-caption dataset
* Add CIDEr, METEOR, and ROUGE-L evaluation
* Add attention mechanism
* Add transformer-based decoder
* Add pretrained vision-language models
* Add experiment tracking
* Add richer dataset analysis scripts
* Add data statement
* Add Streamlit or Gradio demo app
* Add Streamlit or Gradio demo app
* Add Docker support
* Add notebook walkthrough

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* pandas
* NumPy
* Pillow
* NLTK
* Matplotlib
* tqdm
* pytest

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License.

This project is intended for educational and portfolio purposes. If you use or modify it, keep the dataset limitations and model limitations clear.

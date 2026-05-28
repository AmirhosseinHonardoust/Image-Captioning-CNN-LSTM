<div align="center">

# Image Captioning with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN--LSTM-ee4c2c)
![Computer Vision](https://img.shields.io/badge/Task-Image%20Captioning-purple)
![NLP](https://img.shields.io/badge/Decoder-LSTM-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/github/license/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM)

</div>

A PyTorch-based image captioning project that combines a convolutional image encoder with an LSTM language decoder to generate natural-language descriptions for images.

The model follows a classic encoder-decoder architecture:

```text
Image → CNN Encoder → Image Feature Vector → LSTM Decoder → Generated Caption
```

This repository is designed as a clean, educational, and extensible baseline for image captioning experiments.

---

## Overview

Image captioning is a multimodal deep learning task that connects computer vision and natural language processing. Given an input image, the model learns to generate a sequence of words describing the image content.

This project uses:

* **ResNet-50** as the CNN image encoder
* **LSTM** as the text decoder
* **PyTorch** for model development and training
* **BLEU-1 to BLEU-4** for validation and test evaluation
* **Greedy decoding** and **Beam Search** for inference

---

## Features

* CNN-LSTM encoder-decoder architecture
* ResNet-50 image feature extraction
* LSTM-based caption generation
* Custom vocabulary builder
* Train / validation / test split support
* BLEU-1, BLEU-2, BLEU-3, and BLEU-4 evaluation
* Greedy decoding and beam search inference
* Training and validation loss visualization
* BLEU score visualization
* Unit tests for core functionality
* Clean project structure for experimentation

---

## Project Structure

```text
Image-Captioning-CNN-LSTM/
├── data/
│   ├── captions.csv
│   └── images/
│       ├── example1.jpg
│       └── example2.jpg
├── outputs/
│   ├── bleu_scores.png
│   ├── training_curves.png
│   ├── metrics.json
│   └── vocab.json
├── src/
│   ├── infer.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_core.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── LICENSE
```

---

## Model Architecture

### Encoder

The encoder uses a pretrained ResNet-50 backbone to extract visual features from the input image. The final classification layer is removed, and the extracted feature vector is projected into the embedding space used by the decoder.

```text
Input Image → ResNet-50 → Linear Projection → Image Embedding
```

### Decoder

The decoder is an LSTM language model that generates captions word by word. During training, it learns to predict the next word in a caption sequence given the image representation and previous words.

```text
Image Embedding + Caption Tokens → LSTM → Vocabulary Distribution
```

During inference, the model can generate captions using:

* **Greedy decoding**
* **Beam search decoding**

---

## Dataset Format

The training script expects a CSV file with the following columns:

```csv
image_path,caption,split
images/example1.jpg,a blue square with the word example1,train
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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Image-Captioning-CNN-LSTM.git
cd Image-Captioning-CNN-LSTM
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

---

## Run Tests

To verify the core functionality:

```bash
pytest
```

Expected result:

```text
4 passed
```

---

## Training

Train the model using:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --min-freq 1 --num-workers 0
```

Important arguments:

| Argument           | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `--captions`       | Path to the captions CSV file                           |
| `--images-root`    | Root directory used to resolve image paths              |
| `--epochs`         | Number of training epochs                               |
| `--batch-size`     | Training batch size                                     |
| `--min-freq`       | Minimum word frequency required to enter the vocabulary |
| `--num-workers`    | DataLoader worker count                                 |
| `--no-pretrained`  | Disable pretrained ResNet weights                       |
| `--train-backbone` | Fine-tune the CNN backbone                              |

Example with a larger batch size:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --batch-size 16 --min-freq 1 --num-workers 0
```

If pretrained ResNet weights cannot be downloaded, run:

```bash
python src/train.py --captions data/captions.csv --images-root data --epochs 20 --min-freq 1 --num-workers 0 --no-pretrained
```

After training, the following files are generated in `outputs/`:

```text
outputs/best_captioner.pt
outputs/vocab.json
outputs/metrics.json
outputs/training_curves.png
outputs/bleu_scores.png
```

---

## Inference

Run inference on a single image:

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

## Demo Result

The included tiny demo dataset is only intended to verify that the full pipeline works:

```text
dataset → training → checkpoint → inference → generated caption
```

Example outputs from the demo dataset:

```text
example1.jpg → a blue image used for training
example2.jpg → an orange image used for training
```

These outputs confirm that the model can train and generate captions end-to-end.

However, the included demo dataset is extremely small and should not be used to judge real-world image-captioning performance.

---

## Evaluation

The training script reports:

* Training loss
* Validation loss
* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* Test BLEU score when a test split is available

Example log:

```text
[epoch 1] train_loss=1.2785 val_loss=0.7448 BLEU-1=0.3250 BLEU-4=0.2638
[OK] Training done. Best validation BLEU-4: 0.2638
[OK] Test BLEU-4: 0.1792
```

Generated plots:

```text
outputs/training_curves.png
outputs/bleu_scores.png
```

### Important Note About Small Datasets

When training on only a few images, the model will overfit quickly. This usually appears as:

```text
training loss decreases
validation loss increases
BLEU scores fluctuate heavily
```

This behavior is expected for tiny datasets. For meaningful evaluation, train the model on a larger image-caption dataset.

---

## Recommended Dataset Size

For real image-captioning results, use a dataset with many images and diverse captions.

Suggested minimums:

| Dataset Size     | Use Case                    |
| ---------------- | --------------------------- |
| 2–10 images      | Pipeline testing only       |
| 50–100 images    | Small demo                  |
| 500–1,000 images | Better experimental results |
| 5,000+ images    | More realistic training     |

For better performance, each image should ideally have multiple captions.

Example:

```csv
image_path,caption,split
images/cat1.jpg,a cat sitting on a sofa,train
images/cat1.jpg,a small cat resting indoors,train
images/cat1.jpg,a grey cat sitting on furniture,train
```

---

## Limitations

This project is a CNN-LSTM baseline and is intended for learning, experimentation, and portfolio demonstration.

Current limitations:

* The included dataset is tiny and only suitable for testing the pipeline.
* CNN-LSTM models are weaker than modern transformer-based captioning models.
* BLEU scores on very small validation sets are unstable.
* The model requires a larger dataset to generate meaningful real-world captions.
* The checkpoint file is generated after training and is not expected to be committed to GitHub.

---

## Future Improvements

Possible next steps:

* Train on a larger custom dataset
* Add support for multiple reference captions per image
* Add CIDEr, METEOR, and ROUGE-L evaluation
* Add attention mechanism
* Add transformer-based decoder
* Add pretrained vision-language models
* Add experiment tracking
* Add GitHub Actions for automated testing
* Add sample prediction tables to the README

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* Pandas
* NumPy
* NLTK
* Pillow
* Matplotlib
* Pytest

---

## License

This project is licensed under the MIT License.

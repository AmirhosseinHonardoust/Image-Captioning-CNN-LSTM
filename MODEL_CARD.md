# Model Card: CNN-LSTM Image Captioning Baseline

## Model Summary

This repository implements an educational image-captioning baseline with a ResNet-50 CNN encoder and an LSTM text decoder. It is intended for learning, experimentation, and portfolio demonstration rather than production deployment.

## Intended Use

- Demonstrate an end-to-end multimodal ML pipeline.
- Train and evaluate a classic encoder-decoder captioning model.
- Provide a baseline for future improvements such as attention or transformer decoders.

## Not Intended For

- Safety-critical visual interpretation.
- Accessibility or assistive-captioning tools without stronger validation.
- Medical, legal, security, or production image analysis.
- Claims of real-world captioning accuracy using only the included demo dataset.

## Dataset Notes

The included dataset is intentionally tiny. It is useful for smoke-testing the pipeline, but it does not support meaningful generalization claims. For real evaluation, use a larger image-captioning dataset with multiple reference captions per image.

## Evaluation

The training script reports training loss, validation loss, BLEU-1 through BLEU-4, best epoch, optional test metrics, and sample predictions. BLEU scores are overlap metrics and should be interpreted cautiously, especially on small datasets.

## Known Limitations

- No attention mechanism.
- No transformer decoder.
- No object detector or external knowledge source.
- Limited generalization on tiny datasets.
- Generated captions may be incomplete, biased, or incorrect.

## Recommended Next Steps

- Train on a larger dataset.
- Add attention or transformer-based decoding.
- Add CIDEr, METEOR, and ROUGE-L metrics.
- Add a small demo app for manual qualitative testing.

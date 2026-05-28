Prepared image-caption dataset

This is a tiny demo dataset made from the two uploaded images.
It is only useful for checking that the training/inference pipeline works.
It is not enough to train a real image-captioning model.

Expected project usage:

1. Copy or unzip this folder into your repo root.

2. Train:
   python src/train.py --captions prepared_caption_dataset/captions.csv --images-root prepared_caption_dataset --epochs 20 --min-freq 1 --num-workers 0

3. Test:
   python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image prepared_caption_dataset/images/example1.jpg --max-len 10

4. Beam search:
   python src/infer.py --checkpoint outputs/best_captioner.pt --vocab outputs/vocab.json --image prepared_caption_dataset/images/example2.jpg --max-len 10 --beam-size 3

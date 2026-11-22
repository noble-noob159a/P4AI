# Classical Machine Learning vs. Deep Learning on Animals-10

This project compares hand-crafted computer vision with modern deep learning on
**Animals-10**, a dataset of 26,179 web images in 10 animal classes (dog, horse, elephant,
butterfly, chicken, cat, cow, sheep, spider, squirrel). Every model uses the same protocol:

- one stratified 70/15/15 split (seed 42);
- model selection on the validation set only;
- a single evaluation on the test set, reporting macro-F1 with bootstrap confidence intervals;
- calibration, inference cost and visual explanations on the same images.


## Methods

- **Classical track:** HOG, LBP, HSV colour histograms, and bag-of-visual-words (BoVW) over
  HOG and SIFT descriptors with a spatial pyramid. These are combined into 15 pipelines,
  reduced to 512 dimensions with PCA, and classified with Logistic Regression, an RBF-SVM or
  XGBoost. Each classifier is tuned by random search.
- **Trained deep models:** ConvNeXt-T and ViT-S/16, each trained three ways: from
  scratch, from supervised ImageNet weights, and from self-supervised DINOv3 weights.
- **Zero-shot models:**
  - MobileCLIP2-S0, by image–text similarity;
  - Qwen3.5-0.8B, by a generated answer;
  - Valen-Decision-1.5-0.8B, by choice probabilities.
- **Evaluation:**
  - Explanation maps: Grad-CAM, transformer attribution, CLS attention and occlusion.
  - Calibration: ECE, before and after temperature scaling.
  - Cost: latency, FLOPs and parameter counts.

## Key results

| Model | Test macro-F1 |
|---|---|
| Best classical pipeline (SIFT-BoVW + LBP + HSV + HOG, RBF-SVM) | 65.7% |
| ConvNeXt-T / ViT-S/16 from scratch | 67.2% / 45.1% |
| ConvNeXt-T fine-tuned (ImageNet-22k / DINOv3) | 98.9% / 97.6% |
| ViT-S/16 fine-tuned (ImageNet-21k / DINOv3) | 98.5% / 98.5% |
| MobileCLIP2-S0, zero-shot | **98.6%** |
| Qwen3.5-0.8B / Valen-Decision-1.5, zero-shot | 97.6% / 97.6% |

- **What the model was pretrained on matters far more than the architecture or the classifier.**
  Hand-crafted features plateau at about 66%, partly by relying on colour and background cues.
  Pretrained models reach about 98–99%.
- **Self-supervised DINOv3 pretraining, which uses no labels, matches supervised
  pretraining for the ViT.** For ConvNeXt it trails by about 1.3 points.
- **A small CLIP model with no labelled training images ties the best fine-tuned model.** The
  0.8B vision-language models are close behind but 20–30× slower.
- **Fine-tuned models are under-confident because of label smoothing.** A single temperature
  fixes this, lowering ECE from about 0.08 to 0.008 or less. The zero-shot models are
  well calibrated as they are.
- **The explanation maps differ between pretrained and scratch models.** Pretrained models
  focus on the animal, while the scratch CNN often relies on the background.

## Limitations

- Each model was trained once (one seed, one split), and the hyperparameter search budget
  was small.
- The deep models were trained without data augmentation.
- About 2% of the labels are noisy and some test images are near-duplicates of each other,
  so the top models fall within noise of each other.
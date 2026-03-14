# contextual-deepfake-detection
Beyond Binary Classification: Interpretable and Context Aware Deepfake Detection


### Install dependencies

`pip install requirements.txt`

## Dataset Setup

This project uses the **OpenFake dataset**, which contains real and AI-generated images with associated prompts and metadata for evaluating deepfake detection.

Dataset: https://huggingface.co/datasets/ComplexDataLab/OpenFake  

Paper: Livernoche et al., *OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection* ([arXiv](https://arxiv.org/abs/2509.09495), 2025).

### Download dataset

`python3 initialize_dataset.py --sample_size 100`

The dataset will be downloaded and stored locally in:

```
contextual-deepfake-detection/data/
```

> Note: Do not upload the dataset to the repo as it is very large. It is currently ignored in `.gitignore` 

This project explores the use of transformer-based architectures for
binary classification of real vs. fake images. The current implementation
uses a pretrained Vision Transformer (ViT-B/16) fine-tuned on a labeled
dataset of real and manipulated images. Future implementations will use a
hand crafted visual transformer. Main library is pythorch.
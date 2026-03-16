# contextual-deepfake-detection
Beyond Binary Classification: Interpretable and Context Aware Deepfake Detection

This project explores transformer-based architectures for detecting AI-generated images.
The current baseline uses a pretrained Vision Transformer (ViT-B/16) fine-tuned to classify images as real or fake.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Install dependencies
`pip install -r requirements.txt`

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# GPU usage verification (Optional but Reccommended)
If you want to train using a GPU, install the CUDA compatible PyTorch version.

Update/install pytorch if needed (older versions may not support GPU usage): 
    `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

Verify your GPU is available for use when needed:
    `print(torch.cuda.is_available())` true = available, false = unavailable

If unavailable, your system/specs might not support it (CPU will be used). 

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Dataset Setup
This project uses the **OpenFake dataset**, which contains real and AI-generated images with associated prompts and metadata for evaluating deepfake detection.

Dataset: https://huggingface.co/datasets/ComplexDataLab/OpenFake  

Paper: Livernoche et al., *OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection* ([arXiv](https://arxiv.org/abs/2509.09495), 2025).

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Download dataset
`python initialize_dataset.py --sample_size 100`

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

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Project Structure
contextual-deepfake-detection/

Contextual_Deepfake_Detector.py   # main training script
initialize_dataset.py             # dataset download script
requirements.txt

data/
    openfake/

models/
    checkpoints saved during training

results/
    experiment outputs and metrics

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Running Training

Basic run:
`python Contextual_Deepfake_Detector.py`

Custom run:
`python Contextual_Deepfake_Detector.py --epochs 10 --batch_size 32 --lr 0.0001`

Arguments:
- --epochs: Number of training epochs
- --batch_size: Training batch size
- --lr: Learning rate

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Training Pipeline

1. Load the OpenFake dataset from disk
2. Convert dataset into PyTorch dataloaders
3. Split training data into train + validation sets
4. Load pretrained Vision Transformer (ViT-B/16)
5. Replace classification head for binary classification
6. Train the model on real vs fake images
7. Evaluate performance on validation and test sets
8. Save metrics and trained model

Each experiment starts from pretrained ViT weights and trains independently. Trained models are saved inside the results folder.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Evaluation Metrics

The following metrics are computed: 
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix

Results are automatically saved in: `results/run_xxx/`

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Model Checkpoints and Saving

Checkpoints allow resuming of training incase of an unexpected crash.
Checkpoints are saved after every epoch:
    - Ex: `models/checkpoint_epoch_1.pth`

After training finishes, the trained model weights are saved in the experiment results folder:
    - `results/run_timestamp/baseline_vit.pth`

Reloading a trained model:
    `model = build_model()`
    `model.load_state_dict(torch.load("results/run_xxx/baseline_vit.pth"))`
    `model.eval()`
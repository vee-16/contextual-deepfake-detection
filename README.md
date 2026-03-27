# contextual-deepfake-detection
Beyond Binary Classification: Interpretable and Context Aware Deepfake Detection

This project explores transformer-based architectures for detecting AI-generated images.
The current baseline uses a pretrained Vision Transformer (ViT-B/16) fine-tuned to classify images as real or fake.


---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------

# Run on UMBC HPCF (Optional)

1. SSH into cluser: `ssh <username>@chip.rs.umbc.edu`

Verify with DUO/sms code

2. Move to group storage

- Navigate to class directory: `cd /umbc/class/cmsc475sp26/common/`

- Check existing group: `cd /umbc/class/cmsc475sp26/common/vsinha1_group/`

- Or create a new group `mkdir -p <groupname>`

- Clone repo:
```sh
cd <groupname>
git clone https://github.com/vee-16/contextual-deepfake-detection.git
cd contextual-deepfake-detection
```

3. Request compute node: `srun --gres=gpu:1 --mem=8G --time=01:00:00 --pty bash`

- Jobs must be run on compute nodes, not login nodes due to storage limitations.

- Sanity check: `hostname` (expected output: g20-xx)

4. Load python module and create virtual environment

```sh
module load Python/3.11.5-GCCcore-13.2.0
python -m venv .venv
source .venv/bin/activate
```

5. Redirect Cache

```sh
export HF_HOME=/umbc/class/cmsc475sp26/common/<groupname>/hf_cache
export TORCH_HOME=/umbc/class/cmsc475sp26/common/<groupname>/torch_cache

mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

```
TODO:
- [ ] look into tmux set up for persistent sessions
- [ ] add instructions for ssh vscode client

Continue to install dependencies and run the pipeline

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
evaluate_run.py                   # evaluation for a run
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
- --load_model: Path to a saved model or checkpoint (skips training and runs evaluation/inference)
- --num_saliency: Number of saliency maps to generate and save

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
    - ROC-AUC *To be implemented*
    - Confusion Matrix

Results are saved per run in `results/<run_name>/` (e.g. `results/epochs12_bs32_lr0p0001_172242/`). To recompute and save detailed metrics for a run:  
`python evaluate_run.py --run_dir results/<run_name>`

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Model Checkpoints and Saving

Checkpoints allow resuming of training incase of an unexpected crash.
Checkpoints are saved after every epoch:
    - Ex: `models/checkpoint_epoch_1.pth`

After training finishes, the trained model weights are saved in the experiment results folder:
    - `results/<run_name>/baseline_vit.pth`

Reloading a trained model:
    `model = build_model()`
    `model.load_state_dict(torch.load("results/<run_name>/baseline_vit.pth"))`
    `model.eval()`
    * This can be done automatically from the command line with:
        `python Contextual_Deepfake_Detector.py --load_model "models/checkpoint_epoch_5.pth"` (Trained Model)
        `python Contextual_Deepfake_Detector.py --load_model "results/<run_name>/baseline_vit.pth"` (Final Trained Model)


---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Inference and Saliency Maps

This project supports loading a trained model and generating saliency maps for interpretability.

Using saliency maps is our first strategem for defining what in the image is causing it 
to be labelled as fake or real

TODO: Implement an occlusion map to identify most important region as opposed to specific pixels

---------------------------------------------------------------------------------------------

# Running Inference (No Training)

To load a trained model and skip training:

`python Contextual_Deepfake_Detector.py --load_model "models/checkpoint_epoch_5.pth"`

You can also load a final trained model from a previous run:

`python Contextual_Deepfake_Detector.py --load_model "results/<run_name>/baseline_vit.pth"`

---------------------------------------------------------------------------------------------

# Generating Saliency Maps

To generate and save saliency maps:

`python Contextual_Deepfake_Detector.py --load_model "models/checkpoint_epoch_5.pth" --num_saliency 10`

This will:
1. Load the trained model
2. Run evaluation on the test set
3. Generate saliency maps for a subset of test images
4. Save visualizations to: results/<run_name>/saliency_maps/

Each saved image includes:
- Original image
- Saliency heatmap
- True label and predicted label
- Model confidence scores

---------------------------------------------------------------------------------------------

# Notes on Saliency Maps

- Saliency maps are computed using input gradients
- Higher intensity regions indicate stronger influence on the model’s decision
- We will implement an occlusion map using a similar technique to highlight the most
  important area(s)

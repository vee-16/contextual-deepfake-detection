# contextual-deepfake-detection
Beyond Binary Classification: Interpretable and Context Aware Deepfake Detection

This project explores transformer-based architectures for detecting AI-generated images.
The current baseline uses a pretrained Vision Transformer (ViT-B/16) fine-tuned to classify images as real or fake.


---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Project Structure
contextual-deepfake-detection/

Contextual_Deepfake_Detector.py   # main training script
context_reasoning.py              # stage-3 CLIP-based contextual scoring module
initialize_dataset.py             # dataset download script
evaluate_run.py                   # evaluation for a run
evaluate_context_reasoner.py      # evaluation for the CLIP context reasoner
app.py                            # Streamlit demo (upload an image, see the truth)
check_gpu.py                      # GPU usage verification
requirements.txt

```
data/
└── openfake/
    ├── train/
    │   ├── chunk_5000/
    │   ├── chunk_10000/
    │   └── ...
    └── test/
        ├── chunk_5000/
        └── ...
```
Each chunk contains ~5000 samples.  
This design prevents out-of-memory errors when working with large datasets.
```
results/
└── vit_e5_bs32_lr0p0001_20260420_180416/
    ├── metrics.csv
    ├── predictions/
    │   ├── test_y_true.npy
    │   ├── test_y_pred.npy
    │   └── test_y_prob.npy
    ├── saliency_maps/
    │   ├── sample_1.png
    │   └── ...
    ├── occlusion_maps/
    │   ├── sample_1.png
    │   └── ...
    └── model/
        └── checkpoint_epoch_5.pth
```
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Training Pipeline

1. Load chunked OpenFake dataset from disk
2. Convert dataset into PyTorch dataloaders
3. Split training data into train + validation sets
4. Load pretrained Vision Transformer (ViT-B/16)
5. Replace classification head for binary classification
6. Train the model on real vs fake images
7. Evaluate performance on validation set during training
8. Save best model based on validation accuracy
9. Generate predictions on the test set
10. Run evaluation script to compute final metrics

Each experiment starts from pretrained ViT weights and trains independently. Trained models are saved inside the results folder.

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

3. Request compute node

### Dataset Download (CPU node)
`srun --cluster=chip-cpu --partition=general --mem=64G --time=06:00:00 --pty bash`

### Training (GPU node)
`srun --cluster=chip-gpu --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00 --pty bash`

- CPU nodes are recommended for dataset downloading. GPU nodes should be used for training.

- Jobs must be run on compute nodes, not login nodes due to storage limitations.

- Sanity check: `hostname` (expected output: g20-xx)

4. Load python module and create virtual environment

```sh
module load Python/3.11.5-GCCcore-13.2.0
python -m venv .venv
source .venv/bin/activate
```

5. Redirect Cache (Required for HPC)

```sh
export HF_HOME=/umbc/class/cmsc475sp26/common/vsinha1_group/hf_cache
export TORCH_HOME=/umbc/class/cmsc475sp26/common/vsinha1_group/torch_cache
export MPLCONFIGDIR=/umbc/class/cmsc475sp26/common/vsinha1_group/mpl_cache
export PIP_CACHE_DIR=/umbc/class/cmsc475sp26/common/vsinha1_group/pip_cache

mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $MPLCONFIGDIR
mkdir -p $PIP_CACHE_DIR

```
- HF_HOME: HuggingFace cache (dataset + models)
- TORCH_HOME: PyTorch model cache
- MPLCONFIGDIR: Fixes matplotlib "No space left on device" errors on HPC
- PIP_CACHE_DIR: Prevents pip install issues

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

Quick check that PyTorch sees and can use the GPU
    `python check_gpu.py`

If unavailable, your system/specs might not support it (CPU will be used). 

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Dataset Setup
This project uses the **OpenFake dataset**, which contains real and AI-generated images with associated prompts and metadata for evaluating deepfake detection.

Dataset: https://huggingface.co/datasets/ComplexDataLab/OpenFake  

Paper: Livernoche et al., *OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection* ([arXiv](https://arxiv.org/abs/2509.09495), 2025).

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Chunked Dataset Loading

Instead of loading a single dataset file, the pipeline:

- Loads each chunk individually using `load_from_disk`
- Combines all chunks using `concatenate_datasets`
- Treats the result as a single dataset for training

This allows scaling to large datasets without memory issues.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------
# Download dataset

Run first (speeds up downloads):
`export HF_TOKEN=<token>`

Command for dataset download (change sample size as needed):
`python initialize_dataset.py --sample_size 100`

The dataset will be downloaded and stored locally in:

```
contextual-deepfake-detection/data/
```

> Note: Do not upload the dataset to the repo as it is very large. It is currently ignored in `.gitignore` 
> Note: The test split uses all available samples (~59k), while the training split is user-defined.

This project explores the use of transformer-based architectures for
binary classification of real vs. fake images. The current implementation
uses a pretrained Vision Transformer (ViT-B/16) fine-tuned on a labeled
dataset of real and manipulated images. Future implementations will use a
hand crafted visual transformer. Main library is PyTorch.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Running Training

Basic run:
`python Contextual_Deepfake_Detector.py`

Custom run:
`python Contextual_Deepfake_Detector.py --epochs 5 --batch_size 64 --lr 0.0001 --num_saliency 5 --num_occlusion 5`

Arguments:
- --epochs: Number of training epochs
- --batch_size: Training batch size
- --lr: Learning rate
- --load_model: Path to a saved model or checkpoint (skips training and runs evaluation/inference)
- --num_saliency: Number of saliency maps to generate and save

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Evaluation Metrics

Evaluation is performed using a separate script after predictions are saved.

The following metrics are computed:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix
    - Per-score ROC-AUC for perplexity, CLS entropy, rollout, and CLIP context score
    - Combined anomaly score (weighted sum of attention-based scores)

After training or inference completes, the evaluation script runs automatically.

You can also manually run evaluation:

python evaluate_run.py --run_dir results/<run_name>

Evaluation uses saved predictions:
    results/<run_name>/predictions/
        test_y_true.npy
        test_y_pred.npy
        test_y_prob.npy
        test_perplexity.npy
        test_cls_entropy.npy
        test_rollout.npy
        test_context_score.npy   (optional, see CLIP Context Reasoning section)

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# CLIP Context Reasoning

Stage-3 module that scores each image against typical/atypical text prompts using a pretrained
CLIP model (`openai/clip-vit-base-patch32`). Produces a context score in [0, 1] where higher
means more atypical (more likely fake).

Run after training to score the test set and write `test_context_score.npy` into the run's
predictions folder so `evaluate_run.py` can include it:

`python evaluate_context_reasoner.py --save_to_run_dir results/<run_name>`

Then re-run evaluation to pick up the new score:

`python evaluate_run.py --run_dir results/<run_name>`

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Streamlit Demo

Interactive demo that lets you upload an image and see classifier prediction, saliency map,
occlusion map, and CLIP contextual scoring with top-firing prompts.

`streamlit run app.py -- --checkpoint results/<run_name>/best_model.pth`

The `--` is required so Streamlit forwards the `--checkpoint` flag to the script.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Model Saving

During training, the model with the best validation accuracy is saved:

results/<run_name>/best_model.pth

This ensures the best-performing model is used for evaluation and inference.

---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------   
# Inference and Saliency Maps

This project supports loading a trained model and generating saliency maps for interpretability.

Using saliency maps is our first strategy for defining what in the image is causing it 
to be labelled as fake or real

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Running Inference (No Training)

To load a trained model and skip training:

`python Contextual_Deepfake_Detector.py --load_model results/<run_name>/best_model.pth"`

Example:
`python Contextual_Deepfake_Detector.py --load_model results/vit_e5_bs32_lr0p0001_20260329_104307/best_model.pth`

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Generating Saliency Maps

To generate and save saliency maps:

`python Contextual_Deepfake_Detector.py --load_model "models/checkpoint_epoch_5.pth" --num_saliency 10`

This will:
1. Load the trained model
2. Run model on the test set to generate predictions
3. Generate saliency maps for a subset of test images
4. Save visualizations to: results/<run_name>/saliency_maps/

Each saved image includes:
- Original image
- Saliency heatmap
- True label and predicted label
- Model confidence scores

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Notes on Saliency Maps

- Saliency maps are computed using input gradients
- Higher intensity regions indicate stronger influence on the model’s decision
- We will implement an occlusion map using a similar technique to highlight the most
  important area(s)

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Occlusion Maps and Important Regions

In addition to saliency maps, this project implements **occlusion-based interpretability**.

Instead of looking at gradients, occlusion maps systematically cover (mask) parts of the image
and measure how much the model’s prediction changes. This allows us to identify the **most
important region** for the final classification.

For each image, two outputs are generated:
1. **Occlusion Map** (heatmap)
2. **Occlusion Box** (single most important region outlined on the original image)

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Generating Occlusion Maps

To generate occlusion maps, use the following arguments:

- `--num_occlusion`: Number of images to process
- `--occlusion_patch`: Size of the square patch (fixed mode)
- `--occlusion_stride`: Step size for sliding window
- `--use_auto_patch_size`: Automatically select best patch size
- `--occlusion_box_color`: Color of the outlined box (default: black)

Outputs are saved to:
results/<run_name>/occlusion_maps/
results/<run_name>/occlusion_boxes/

Each result includes:
- Original image
- Occlusion heatmap
- Highest-impact region outlined on the original image
- Model predictions and probabilities

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Example Commands (Occlusion)

### Fixed patch size
Uses a constant patch size and stride (faster, more predictable):

`python Contextual_Deepfake_Detector.py --load_model "models/checkpoint_epoch_5.pth"
--num_occlusion 5
--occlusion_patch 32
--occlusion_stride 16
--occlusion_box_color black`

### Auto patch size
Automatically selects the patch size that produces the strongest signal (slower, better visualization):

`python Contextual_Deepfake_Detector.py
--load_model "models/checkpoint_epoch_5.pth"
--num_occlusion 5
--use_auto_patch_size
--occlusion_box_color black`

# Notes on Occlusion Maps

- Occlusion maps highlight **regions**, not individual pixels
- The outlined box represents the area that causes the **largest drop in model confidence**
- Fixed patch mode is faster and easier to debug
- Auto patch mode produces clearer and more interpretable results

---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 
# Example Commands

**Training**
`python Contextual_Deepfake_Detector.py --epochs 5 --batch_size 32 --lr 0.0001`

**Saliency Maps (custom number)**
`python Contextual_Deepfake_Detector.py --load_model results/<run_name>/best_model.pth --num_saliency 10`

**Occlusion maps (fixed patch)**
`python Contextual_Deepfake_Detector.py --load_model results/<run_name>/best_model.pth --num_occlusion 5 --occlusion_patch 32 --occlusion_stride 16 --occlusion_box_color black`

**Occlusion maps (auto patch selection)**
`python Contextual_Deepfake_Detector.py --load_model results/<run_name>/best_model.pth --num_occlusion 5 --use_auto_patch_size --occlusion_box_color black`

**Full training and Visualization**
`python Contextual_Deepfake_Detector.py --epochs 5 --batch_size 32 --lr 0.0001 --num_saliency 5 --num_occlusion 5 --use_auto_patch_size`

**CLIP context evaluation (after training)**
`python evaluate_context_reasoner.py --save_to_run_dir results/<run_name>`

**Streamlit demo**
`streamlit run app.py -- --checkpoint results/<run_name>/best_model.pth`

---------------------------------------------------------------------------------------------

# Single Image Explanation
(results/demo_explanation/explanation_report.png)
This project now includes a single-image explanation pipeline that generates a comprehensive interpretability report for any input image.

This feature combines multiple signals:

Model prediction (real vs fake) with confidence
Saliency map (pixel-level importance)
Occlusion map + bounding box (region-level importance)
Attention rollout heatmap (transformer-level reasoning)
CLIP context reasoning score (semantic consistency)

Run Explanation on One Image:

python explain_image.py \
  --image path/to/image.jpg \
  --checkpoint results/<run_name>/best_model.pth \
  --output_dir results/demo_explanation

The script generates:

results/demo_explanation/
├── explanation_report.png   # Visual summary (demo-ready)
└── explanation_report.json  # Raw scores and metadata

Explanation Report Includes:

Prediction: real or fake
Confidence: model probability
Saliency Map: pixel-level influence
Occlusion Map: region importance + bounding box
Attention Rollout: patch-level influence across transformer layers
CLIP Context Score: measures how "typical" or "atypical" the image is 
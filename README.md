# RT-DETR Railway Obstacle Detection with Segmentation-Guided Attention

A research implementation of **RAAG-DETR** (Rail-Aware Attention Gating for RT-DETR), which adds a novel segmentation-guided attention mechanism to RT-DETR for real-time railway obstacle detection.

## Overview

This project addresses a critical limitation in applying general-purpose object detectors to railway safety: **spatially uniform attention**. Standard RT-DETR treats all image regions equally, leading to false positives in background areas irrelevant to railway operations. Our approach introduces a lightweight rail corridor segmentation head that guides transformer attention to focus on safety-critical regions.

## Research Contribution

**RAAG-DETR** introduces a novel algorithmic component: **segmentation-guided spatial attention biasing** for transformer-based detectors. Unlike multi-task learning approaches where segmentation and detection are independent parallel tasks, we use segmentation predictions to explicitly modulate attention logits in RT-DETR's transformer encoder, creating a differentiable gating mechanism.

### Key Features

1. **Lightweight Rail Segmentation Head**: Adds < 1M parameters to predict binary rail corridor masks
2. **Shared Backbone**: Segmentation and detection share the same feature extractor (no redundant computation)
3. **Attention Gating**: Rail masks are converted to logit-space biases and injected into transformer attention
4. **Real-Time Performance**: Maintains RT-DETR's edge-deployable speed (~30 FPS) with minimal overhead
5. **End-to-End Training**: Joint optimization of detection and segmentation with combined loss function

### Novelty

This is **not** standard RT-DETR (which lacks spatial attention priors) nor standard YOLO (which has no transformer attention to modulate). The novelty lies in:

- **Functional coupling** between segmentation and attention (not just auxiliary task learning)
- **Logit-space bias injection** rather than hard masking (preserves gradient flow)
- **Minimal architectural overhead** while adding domain-specific awareness

See [ALGORITHM.md](ALGORITHM.md) for detailed algorithm description, step-by-step process, pseudo-code, and research explanation.

## Dataset

We use the **RailFOD23** (Railway Foreign Object Detection 2023) dataset for training and evaluation.

**Download Link**: [RailFOD23 on Figshare](https://figshare.com/articles/figure/RailFOD23_zip/24180738?file=43616139)

### Dataset Information

- **Purpose**: Railway obstacle detection in real-world conditions
- **Contents**: Images of railway tracks with labeled obstacles and rail corridor annotations
- **Format**: Standard object detection format (COCO/YOLO compatible)
- **Size**: ~[To be determined after download]

### Download Instructions

```bash
# Create data directory
mkdir -p data

# Download dataset (manual download required from Figshare)
# Navigate to: https://figshare.com/articles/figure/RailFOD23_zip/24180738?file=43616139
# Download RailFOD23.zip and place in data/ directory

# Extract dataset
cd data
unzip RailFOD23.zip
cd ..
```

**Note**: For rail corridor annotations (required for segmentation head training), you may need to:
1. Use existing rail segmentation datasets, or
2. Generate rail masks from bounding box annotations, or
3. Create annotations using rail line detection algorithms as pseudo-labels

## Project Structure

```
.
├── ALGORITHM.md              # Detailed algorithm description and pseudo-code
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory (created after download)
│   └── RailFOD23/           # Extracted dataset
├── models/                   # Model implementations (to be created)
│   ├── raag_detr.py         # RAAG-DETR main model
│   ├── seg_head.py          # Lightweight segmentation head
│   └── rail_attention.py    # Rail-aware attention mechanism
├── train.py                  # Training script (to be created)
├── inference.py              # Inference script (to be created)
└── utils/                    # Utility functions (to be created)
    ├── data_loader.py       # Dataset loading and preprocessing
    └── metrics.py           # Evaluation metrics
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- PyTorch 2.0+

### Setup

```bash
# Clone repository
git clone https://github.com/badrinagarjun/aws-demo.git
cd aws-demo

# Install dependencies
pip install -r requirements.txt

# Note: RT-DETR integration requires the RT-DETR codebase
# Install from official repository or implement following IMPLEMENTATION.md

# Download and extract dataset (see Dataset section above)
```

**Note**: This repository contains the algorithm design and implementation guide. The actual model implementation (train.py, models/, etc.) should be created following the structure described in IMPLEMENTATION.md.

## Usage

**Note**: The following commands are illustrative examples showing how the system should be used once implemented. See IMPLEMENTATION.md for guidance on creating these scripts.

### Training

```bash
# Train RAAG-DETR on RailFOD23 dataset (once train.py is implemented)
python train.py \
    --data data/RailFOD23 \
    --backbone resnet50 \
    --epochs 100 \
    --batch-size 16 \
    --seg-loss-weight 0.3 \
    --rail-lambda 1.0
```

**Key Hyperparameters**:
- `--seg-loss-weight`: Weight for segmentation loss in multi-task learning (β in paper, suggested: 0.1-0.5)
- `--rail-lambda`: Attention gating strength (λ in paper, suggested: 0.5-2.0)

### Inference

```bash
# Run inference on test images
python inference.py \
    --model checkpoints/raag_detr_best.pth \
    --input test_images/ \
    --output results/ \
    --visualize
```

### Evaluation

```bash
# Evaluate on validation set
python evaluate.py \
    --model checkpoints/raag_detr_best.pth \
    --data data/RailFOD23/val
```

## Algorithm Summary

See [ALGORITHM.md](ALGORITHM.md) for complete details. Brief overview:

1. **Shared Feature Extraction**: Use RT-DETR backbone (ResNet/similar) for both tasks
2. **Rail Segmentation**: Lightweight head predicts binary rail corridor mask
3. **Logit-Space Bias**: Convert mask to log-odds spatial bias
4. **Attention Gating**: Inject bias into transformer encoder attention logits
5. **Joint Training**: Optimize L_total = L_det + β·L_seg

**Result**: Railway-aware detector that focuses on rail corridor while maintaining real-time performance.

## Expected Results

Based on the algorithm design, we expect:

- **Improved Precision**: Reduced false positives in background regions (non-rail areas)
- **Maintained Recall**: Detection performance on rail corridor preserved or improved
- **Real-Time Performance**: Minimal computational overhead (< 5% inference time increase)
- **Ablation Study**: Attention gating shows clear benefit over baseline RT-DETR

## Implementation Roadmap

- [x] Algorithm design and documentation
- [x] Project structure setup
- [ ] Implement segmentation head (`models/seg_head.py`)
- [ ] Implement rail-aware attention mechanism (`models/rail_attention.py`)
- [ ] Integrate with RT-DETR backbone (`models/raag_detr.py`)
- [ ] Create data loading pipeline (`utils/data_loader.py`)
- [ ] Implement training script with multi-task loss (`train.py`)
- [ ] Add inference and visualization tools (`inference.py`)
- [ ] Evaluate and benchmark performance
- [ ] Write research paper/technical report

## Research Questions

This implementation enables investigation of:

1. **Gating Strength**: How does λ (rail-aware bias weight) affect precision-recall trade-off?
2. **Segmentation Quality**: How sensitive is detection to segmentation accuracy?
3. **Attention Visualization**: Where does the network attend with vs. without rail gating?
4. **Generalization**: Does rail-aware attention transfer to other linear infrastructure (highways, pipelines)?

## Citation

If you use this code or algorithm in your research, please cite:

```bibtex
@misc{raagdetr2024,
  title={RAAG-DETR: Rail-Aware Attention Gating for Real-Time Railway Obstacle Detection},
  author={[Your Name]},
  year={2024},
  note={Research implementation of segmentation-guided attention for RT-DETR}
}
```

## References

- **RT-DETR**: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- **DETR**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **RailFOD23**: [Railway Foreign Object Detection Dataset](https://figshare.com/articles/figure/RailFOD23_zip/24180738)

## License

This research implementation is provided for academic and research purposes. Please check the licenses of the underlying components (RT-DETR, dataset) for commercial use restrictions.

## Acknowledgments

- RT-DETR authors for the base detector architecture
- RailFOD23 dataset creators for providing high-quality railway detection data
- Railway safety research community for domain expertise

## Contact

For questions about the algorithm or implementation:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This is a research-oriented implementation focused on algorithmic novelty. For production railway safety systems, additional engineering (redundancy, failsafes, certification) is required.

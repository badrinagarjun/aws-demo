# RAAG-DETR Implementation Guide

This document provides implementation details and guidance for building the RAAG-DETR system.

## Architecture Overview

```
Input Image (Railway Scene)
    ↓
[Shared Backbone: ResNet-50/HRNet]
    ↓
    ├─────────────────────────────────┬──────────────────────────────┐
    ↓                                 ↓                              ↓
[Lightweight Segmentation Head]  [Transformer Encoder]    [RT-DETR Decoder]
    ↓                                 ↑                              ↓
Binary Rail Mask ──────────→ Rail-Aware Attention Gating      Object Detections
    ↓                                                               ↓
Segmentation Loss                                          Detection Loss
    └────────────────────────────┬───────────────────────────────┘
                                 ↓
                          Combined Loss (L_total)
```

## Core Components

### 1. Shared Backbone

**Purpose**: Extract multi-scale visual features for both segmentation and detection

**Implementation Notes**:
- Use pre-trained ResNet-50 or HRNet from torchvision/timm
- Extract features at multiple scales (e.g., C3, C4, C5)
- Freeze early layers during initial training for stability
- Fine-tune entire backbone after convergence

**Code Structure**:
```python
# models/backbone.py
class SharedBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        # Load pre-trained backbone
        # Return multi-scale features
```

### 2. Lightweight Segmentation Head

**Purpose**: Predict binary rail corridor mask with minimal computation

**Architecture**:
- Input: High-level backbone features (e.g., C5 with 2048 channels)
- Conv layer: 2048 → 256 channels (3×3, ReLU)
- Upsampling: 4× bilinear interpolation
- Output conv: 256 → 1 channel (1×1, Sigmoid)
- Total parameters: < 1M

**Design Rationale**:
- Lightweight to preserve real-time performance
- Shares backbone features (no separate encoder)
- Binary output (rail vs. non-rail) simplifies task

**Code Structure**:
```python
# models/seg_head.py
class RailSegmentationHead(nn.Module):
    def __init__(self, in_channels=2048, hidden_dim=256, upsample_factor=4):
        # Lightweight decoder
        # Binary mask prediction
```

### 3. Rail-Aware Attention Gating

**Purpose**: Modulate transformer attention based on predicted rail mask

**Mathematical Formulation**:

Standard attention:
```
A = softmax(QK^T / √d_k) V
```

Rail-aware attention:
```
M' = downsample(M, size=(h, w))           # Rail mask at feature resolution
B = log(M' + ε) - log(1 - M' + ε)         # Logit-space bias
A_logits = QK^T / √d_k + λ · B            # Bias injection
A = softmax(A_logits) V                   # Gated attention
```

**Key Parameters**:
- λ (lambda): Gating strength (learnable or fixed, range: 0.5-2.0)
- ε (epsilon): Numerical stability constant (1e-6)

**Implementation Notes**:
- Apply to encoder self-attention layers
- Optionally apply to decoder cross-attention
- Broadcasting: B is (B, h×w, 1), added to attention logits (B, h×w, h×w)
- Preserves gradient flow (differentiable throughout)

**Code Structure**:
```python
# models/rail_attention.py
class RailAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads, rail_lambda=1.0):
        # Multi-head attention with rail bias
        # Learnable or fixed lambda
        
    def forward(self, query, key, value, rail_mask):
        # Compute attention logits
        # Apply rail bias
        # Return gated attention output
```

### 4. RAAG-DETR Integration

**Purpose**: Combine all components into end-to-end trainable model

**Forward Pass**:
1. Extract backbone features: `F = backbone(I)`
2. Predict rail mask: `M = seg_head(F)`
3. Encode with rail-aware attention: `E = encoder(F, M)`
4. Decode to detections: `D = decoder(E)`
5. Return detections and mask: `(D, M)`

**Training Loss**:
```
L_total = L_det + β · L_seg

L_det = L_cls + L_bbox + L_giou  (RT-DETR losses)
L_seg = BCE(M, M_gt)              (Binary cross-entropy)
```

**Code Structure**:
```python
# models/raag_detr.py
class RAAGDETR(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', 
                 seg_loss_weight=0.3, rail_lambda=1.0):
        # Initialize all components
        # Set hyperparameters
        
    def forward(self, images, targets=None):
        # Forward pass through all components
        # Compute losses if training
        # Return predictions
```

## Training Pipeline

### Data Preparation

**Required Annotations**:
1. **Object Detection**: Bounding boxes + class labels (standard COCO format)
2. **Rail Segmentation**: Binary masks for rail corridor regions

**Data Augmentation**:
- Random horizontal flip (50% probability)
- Random brightness/contrast adjustment
- Random scale jittering (0.8-1.2×)
- Random crop and resize
- Normalize to ImageNet statistics

**Code Structure**:
```python
# utils/data_loader.py
class RailFODDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        # Load detection annotations
        # Load segmentation masks
        # Apply transforms
        
    def __getitem__(self, idx):
        # Return: image, bboxes, labels, rail_mask
```

### Training Configuration

**Hyperparameters**:
- Batch size: 16 (adjust based on GPU memory)
- Learning rate: 1e-4 (AdamW optimizer)
- Weight decay: 1e-4
- Epochs: 100-150
- LR schedule: Cosine annealing with warm-up (10 epochs)
- β (seg_loss_weight): 0.3 (tune in range 0.1-0.5)
- λ (rail_lambda): 1.0 (tune in range 0.5-2.0)

**Training Strategy**:
1. **Phase 1** (epochs 1-30): Train with frozen backbone
2. **Phase 2** (epochs 31-100): Fine-tune entire network
3. **Phase 3** (optional): Tune λ for optimal precision-recall balance

**Code Structure**:
```python
# train.py
def train_one_epoch(model, dataloader, optimizer, device):
    # Training loop
    # Compute combined loss
    # Backprop and optimize
    
def validate(model, dataloader, device):
    # Validation loop
    # Compute metrics
    
def main():
    # Setup model, data, optimizer
    # Training loop with validation
    # Save checkpoints
```

### Evaluation Metrics

**Detection Performance**:
- mAP@0.5 (COCO-style)
- mAP@0.5:0.95
- Precision-Recall curves
- False Positive Rate (FPR) in non-rail regions

**Segmentation Performance**:
- IoU (Intersection over Union) for rail mask
- Pixel accuracy
- F1-score

**Runtime Performance**:
- Inference time (ms per image)
- FPS on target hardware
- Model size (parameters, FLOPs)

**Code Structure**:
```python
# utils/metrics.py
def compute_detection_metrics(predictions, targets):
    # mAP, precision, recall
    
def compute_segmentation_metrics(pred_masks, gt_masks):
    # IoU, accuracy, F1
    
def compute_inference_speed(model, dataloader, device):
    # FPS, latency
```

## Inference Pipeline

**Input**: Railway scene image
**Output**: 
- Detected objects (bboxes, classes, confidence scores)
- Rail corridor mask (optional visualization)

**Steps**:
1. Load trained model
2. Preprocess image (resize, normalize)
3. Forward pass through model
4. Post-process detections (NMS, confidence threshold)
5. Optionally filter detections outside rail corridor
6. Visualize results

**Code Structure**:
```python
# inference.py
def load_model(checkpoint_path):
    # Load trained RAAG-DETR
    
def preprocess_image(image_path):
    # Load and preprocess
    
def postprocess_detections(detections, rail_mask, threshold=0.5):
    # NMS, confidence filtering
    # Optional rail-based filtering
    
def visualize_results(image, detections, rail_mask):
    # Draw bounding boxes
    # Overlay rail mask
    # Save or display
```

## Ablation Studies

To validate the novelty and effectiveness of rail-aware attention:

### Experiment 1: Baseline Comparison
- **Baseline**: Standard RT-DETR (no segmentation, no attention gating)
- **RAAG-DETR**: Full model with segmentation and attention gating
- **Metric**: mAP, FPR in non-rail regions

### Experiment 2: Segmentation-Only Multi-Task
- **Variant**: RT-DETR + segmentation head (no attention gating)
- **Purpose**: Isolate benefit of attention gating vs. just multi-task learning
- **Metric**: Detection performance with and without attention gating

### Experiment 3: Gating Strength Sweep
- **Vary λ**: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0
- **Purpose**: Find optimal gating strength
- **Metric**: Precision-Recall trade-off

### Experiment 4: Attention Visualization
- **Method**: Visualize attention maps with and without rail gating
- **Purpose**: Qualitative analysis of where model attends
- **Tool**: GradCAM or attention map visualization

## Expected Challenges and Solutions

### Challenge 1: Rail Mask Quality
**Problem**: Poor segmentation degrades detection
**Solution**: 
- Pre-train segmentation head separately
- Use stronger segmentation backbone initially
- Apply curriculum learning (easy → hard examples)

### Challenge 2: Computational Overhead
**Problem**: Segmentation head + attention gating slows inference
**Solution**:
- Use extremely lightweight segmentation head (< 1M params)
- Shared backbone (no duplicate computation)
- Optimize with TensorRT/ONNX for deployment

### Challenge 3: Hyperparameter Tuning
**Problem**: β and λ require careful tuning
**Solution**:
- Start with β=0.3, λ=1.0 (reasonable defaults)
- Grid search or Bayesian optimization
- Monitor validation metrics closely

### Challenge 4: Dataset Annotation
**Problem**: RailFOD23 may lack rail corridor masks
**Solution**:
- Generate pseudo-labels with classical rail detection
- Use line detection + morphological operations
- Manually annotate subset for validation

## Deployment Considerations

### Edge Device Optimization
- Export to ONNX format
- Quantization (INT8) for faster inference
- TensorRT optimization for NVIDIA devices
- CoreML for iOS deployment

### Real-Time Requirements
- Target: 30+ FPS on edge GPU (Jetson Xavier, etc.)
- Optimize batch processing for video streams
- Asynchronous inference pipeline

### Safety Considerations
**Important**: This is a research implementation. Production railway safety systems require:
- Redundant sensors and detection systems
- Fail-safe mechanisms
- Rigorous testing and certification
- Human oversight and intervention protocols

## Next Steps

1. **Implement Core Components**: Start with models/ directory
2. **Setup Data Pipeline**: Implement utils/data_loader.py
3. **Training Script**: Create train.py with multi-task loss
4. **Validation**: Test on RailFOD23 dataset
5. **Ablation Studies**: Validate novelty and effectiveness
6. **Optimize for Deployment**: Export and optimize model
7. **Documentation**: Write research paper/technical report

## Resources

- **RT-DETR Paper**: https://arxiv.org/abs/2304.08069
- **RT-DETR Code**: https://github.com/lyuwenyu/RT-DETR
- **DETR Tutorial**: https://github.com/facebookresearch/detr
- **RailFOD23 Dataset**: https://figshare.com/articles/figure/RailFOD23_zip/24180738

## Questions?

Open an issue on GitHub or refer to ALGORITHM.md for detailed algorithm description.

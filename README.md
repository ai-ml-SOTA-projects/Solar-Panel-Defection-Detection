# Solar Panel Defect Detection: SOTA Model Comparison

Comparative benchmark of six computer vision architectures for binary classification of solar panel defects (crack vs. hotspot detection).

## Models Evaluated

1. **YOLOv12-XL** - Object detection approach
2. **Qwen 3 VLM-8B** - Vision-language model with LoRA fine-tuning
3. **ConvNeXt V2** - Modern convolutional architecture
4. **ModernCNN** - Custom architecture with dual attention mechanisms
5. **EfficientNet-B0** - Efficient baseline
6. **ResNet50** - Classical baseline
7. **Swin Transformer V2** - Vision transformer architecture

## Dataset

- Training: 1,000+ ultraviolet/thermal images
- Validation: 329 images
- Test: 164 images
- Classes: Crack (0), Hotspot (1)
- Format: Image folder structure with metadata CSV

## Methodology

### Custom Architecture (ModernCNN)

Implemented novel CNN combining:
- Depthwise separable convolutions for parameter efficiency
- Inverted bottleneck blocks (expand-depthwise-compress)
- Dual attention: Squeeze-Excitation (channel) + Coordinate Attention (spatial)
- Progressive downsampling with residual connections

### Fine-tuning Approach

- ConvNeXt V2, Swin V2: HuggingFace Trainer API
- Qwen 3 VLM: Unsloth + LoRA (vision layers, attention, MLP)
- YOLOv12-XL: Ultralytics framework
- PyTorch models: Custom training loops with AdamW, cosine scheduling

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ModernCNN | 1.000 | 1.000 | 1.000 | 1.000 |
| Swin V2 | 1.000 | 1.000 | 1.000 | 1.000 |
| ResNet50 | 1.000 | 1.000 | 1.000 | 1.000 |
| YOLOv12-XL | 0.963 | 0.963 | 0.963 | 0.963 |
| Qwen 3 VLM | 0.762 | 0.573 | 0.762 | 0.653 |
| EfficientNet-B0 | 0.640 | 0.732 | 0.640 | 0.673 |

## Key Findings

1. **CNN architectures achieved perfect accuracy** on this dataset, indicating high visual separability between crack and hotspot defects in ultraviolet imaging.

2. **ModernCNN matched transformer performance** while maintaining significantly lower computational cost and faster inference.

3. **Vision-language models underperformed** for simple binary classification, confirming they are better suited for complex reasoning tasks rather than pattern recognition.

4. **Detection vs. classification tradeoff**: YOLOv12 provides spatial localization at slight accuracy cost compared to pure classification models.

## Technical Stack

- **Frameworks**: PyTorch, HuggingFace Transformers, Ultralytics, Unsloth
- **Training**: Mixed precision (FP16), gradient accumulation, cosine annealing
- **Evaluation**: Scikit-learn metrics, confusion matrices
- **Hardware**: NVIDIA L4 GPU (Google Colab)

## Architecture Details

### ModernCNN Block Structure
```
Input -> LayerNorm -> 1×1 Conv (expand) -> GELU ->
7×7 Depthwise Conv -> GELU → SE Attention ->
Coordinate Attention -> 1×1 Conv (compress) ->
Dropout -> Add Residual -> Output
```

### Training Configuration
- Batch size: 16-32
- Learning rate: 1e-3 to 5e-5 (model-dependent)
- Epochs: 10
- Optimizer: AdamW (weight decay: 0.01)
- Augmentation: Resize, RandomCrop, HorizontalFlip, Normalize

## Files

- Fine_Tuning_YOLOv12,_Qwen_3_VLM,_Building_ConvXNet2,_Training_SwinNet_(Microsoft),_Building_Custom_CNN,_train_ResNet_and_EfficientNet.ipynb

## Future Work

- Multi-class defect classification (cracks, hotspots, corrosion, delamination)
- Instance segmentation with SAM 2
- Deployment optimization (ONNX, TensorRT)
- Real-time inference pipeline for manufacturing

## Author

Kevlar - UC Riverside, Data Science BCOE

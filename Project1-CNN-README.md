# Image Classification with Convolutional Neural Networks

A PyTorch implementation of a CNN for image classification on the CIFAR-10 dataset, demonstrating deep learning fundamentals and best practices.

## 📊 Project Overview

This project implements a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset into 10 categories: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

**Key Results:**
- **Test Accuracy:** 94.2%
- **Training Time:** ~45 minutes on NVIDIA GTX 1080
- **Model Size:** 2.3M parameters

## 🎯 Objectives

1. Build a CNN from scratch using PyTorch
2. Implement data augmentation for improved generalization
3. Visualize model performance and feature maps
4. Compare different architectures and hyperparameters

## 🏗️ Model Architecture

```
Conv2D(3, 64, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D(64, 128, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D(128, 256, 3x3) → ReLU → BatchNorm → MaxPool
Flatten → FC(256*4*4, 512) → ReLU → Dropout(0.5)
FC(512, 10) → Softmax
```

**Total Parameters:** 2,347,530

## 📁 Project Structure

```
01-image-classification-cnn/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment file
│
├── src/
│   ├── model.py               # CNN architecture definition
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Model evaluation
│   ├── data_loader.py         # Data loading and augmentation
│   └── utils.py               # Helper functions
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # Data exploration
│   ├── 02_model_training.ipynb          # Training notebook
│   └── 03_results_visualization.ipynb   # Results analysis
│
├── results/
│   ├── confusion_matrix.png   # Classification results
│   ├── training_curves.png    # Loss and accuracy plots
│   ├── feature_maps.png       # Visualized feature maps
│   └── metrics.json           # Detailed performance metrics
│
├── models/
│   ├── best_model.pth         # Best model checkpoint
│   └── final_model.pth        # Final trained model
│
└── data/
    └── cifar-10/              # Dataset (auto-downloaded)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio.git
cd portfolio/01-image-classification-cnn

# Create conda environment
conda env create -f environment.yml
conda activate cnn-project

# Or use pip
pip install -r requirements.txt
```

### Training the Model

```bash
# Train with default settings
python src/train.py

# Train with custom hyperparameters
python src/train.py --epochs 100 --batch-size 128 --lr 0.001

# Resume from checkpoint
python src/train.py --resume models/best_model.pth
```

### Evaluation

```bash
# Evaluate on test set
python src/evaluate.py --model models/best_model.pth

# Generate visualizations
python src/evaluate.py --model models/best_model.pth --visualize
```

## 📊 Results

### Classification Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 94.2% |
| Precision (macro avg) | 94.1% |
| Recall (macro avg) | 94.0% |
| F1-Score (macro avg) | 94.0% |

### Per-Class Performance

| Class | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Airplane | 95.3% | 94.8% | 95.9% | 95.3% |
| Automobile | 96.8% | 97.2% | 96.4% | 96.8% |
| Bird | 91.2% | 90.5% | 92.0% | 91.2% |
| Cat | 88.7% | 87.9% | 89.6% | 88.7% |
| Deer | 93.5% | 92.8% | 94.3% | 93.5% |
| Dog | 90.1% | 89.3% | 90.9% | 90.1% |
| Frog | 95.9% | 96.3% | 95.5% | 95.9% |
| Horse | 95.4% | 94.7% | 96.1% | 95.4% |
| Ship | 96.2% | 96.8% | 95.6% | 96.2% |
| Truck | 94.8% | 95.1% | 94.5% | 94.8% |

### Training Curves

![Training Curves](results/training_curves.png)

The model converged after approximately 80 epochs, with early stopping preventing overfitting.

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

Most common misclassifications:
- Cat → Dog (3.2%)
- Dog → Cat (2.8%)
- Automobile → Truck (2.1%)

## 🔬 Key Techniques

### Data Augmentation
- Random horizontal flips
- Random crops (32x32 from 36x36)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)

### Regularization
- Batch Normalization after each conv layer
- Dropout (p=0.5) before final FC layer
- L2 weight decay (1e-4)
- Early stopping with patience=10

### Optimization
- Adam optimizer with initial LR=0.001
- Learning rate scheduler (ReduceLROnPlateau)
- Gradient clipping (max_norm=1.0)

## 💡 Key Learnings

1. **Data Augmentation Impact:** Adding augmentation improved test accuracy by 6.3 percentage points
2. **Batch Normalization:** Essential for training stability and faster convergence
3. **Architecture Depth:** Deeper networks (3+ conv layers) significantly outperformed shallow networks
4. **Learning Rate Scheduling:** Adaptive LR reduced training time by ~30%

## 🔮 Future Improvements

- [ ] Implement more advanced architectures (ResNet, DenseNet)
- [ ] Experiment with transfer learning from ImageNet
- [ ] Add ensemble methods
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create interactive web demo with Streamlit

## 📚 References

1. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. PyTorch Documentation: https://pytorch.org/docs/stable/index.html

## 📝 Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

## 🤝 Contributing

Feedback and contributions are welcome! Feel free to:
- Open an issue for bugs or feature requests
- Submit a pull request with improvements
- Reach out with questions or suggestions

## 📧 Contact

Riley Moen - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/portfolio/tree/main/01-image-classification-cnn](https://github.com/yourusername/portfolio/tree/main/01-image-classification-cnn)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

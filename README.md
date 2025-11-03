# African Wildlife Image Analysis Using Computer Vision and Deep Learning

A computer vision project demonstrating the combination of classical image processing techniques and modern deep learning for African wildlife classification. This project analyzes images of buffalo, elephant, rhino, and zebra using OpenCV and TensorFlow, achieving **95.83% overalll classification accuracy**.

## Project Overview

This coursework implements a complete pipeline for wildlife image analysis:

- **Part A: Classical Computer Vision** - Explores OpenCV techniques including grayscale conversion, filtering, edge detection (Canny/Sobel), and contour analysis
- **Part B: Deep Learning Classification** - Uses transfer learning with MobileNetV2 to classify four African wildlife species

### Key Results

| Species | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Elephant | 100% | 1.00 | 1.00 | 1.00 |
| Zebra | 100% | 0.96 | 1.00 | 0.98 |
| Buffalo | 96.67% | 0.88 | 0.97 | 0.92 |
| Rhino | 90% | 1.00 | 0.90 | 0.95 |
| **Overall** | **95.83%** | **0.96** | **0.97** | **0.96** |

## Contents

- `african_wildlife_analysis.ipynb` - Complete Jupyter notebook with all analysis and code
- `BRIEF_REPORT.md` - 4-page executive report on results, challenges, and interpretations
- `PROJECT_REPORT.md` - Comprehensive 10+ page detailed report
- `README.md` - This file

## Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Clone the Repository

```bash
git clone https://github.com/Hansen256/artificial_intelligence_coursework_computer_vision_african_wildlife_analysis.git
cd artificial_intelligence_coursework_computer_vision_african_wildlife_analysis
```

### Install Dependencies

```bash
# Install required packages from requirements.txt
pip install -r requirements.txt
```

This installs:

- TensorFlow (deep learning framework)
- OpenCV (computer vision library)
- NumPy, Pandas, Matplotlib (data processing & visualization)
- Scikit-learn (machine learning metrics)
- Jupyter (notebook environment)
- Kaggle (dataset access)

### Setup Kaggle Authentication

The notebook downloads the African Wildlife dataset from Kaggle. You need to authenticate:

1. **Create a Kaggle account** at https://www.kaggle.com (if you don't have one) <!--markdownlint-disable-line-->

2. **Generate API credentials:**
   - Go to your Kaggle account settings: https://www.kaggle.com/settings/account <!--markdownlint-disable-line-->
   - Click "Create New API Token"
   - This downloads `kaggle.json`

3. **Place `kaggle.json` in the correct location:**

   **On Windows:**

   ```bash
   # Create the .kaggle directory in your user home
   mkdir %USERPROFILE%\.kaggle
   
   # Place kaggle.json in this directory
   # C:\Users\<YourUsername>\.kaggle\kaggle.json
   ```

   **On macOS/Linux:

   ```bash
   mkdir -p ~/.kaggle
   # Place kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions
   ```

4. **Verify setup:**

   ```bash
   kaggle datasets list
   ```

   If this works without errors, you're all set!

### Run the Notebook

```bash
jupyter notebook african_wildlife_analysis.ipynb
```

Then navigate to the notebook in your browser and run cells sequentially (Cell → Run All, or run individual cells with Shift+Enter).

## Project Structure

### Part A: Classical Computer Vision Analysis

The notebook begins with fundamental image processing techniques:

1. **Library Setup** - Import TensorFlow, OpenCV, and related libraries
2. **Dataset Exploration** - Load and explore 1,504 wildlife images (376/class)
3. **Image Preprocessing**
   - Grayscale conversion for comparison with original images
   - Gaussian blur and median filtering for noise reduction
4. **Edge Detection**
   - Canny edge detector for clean, well-defined edges
   - Sobel operators for directional edge information
5. **Contour Analysis** - Identify continuous boundaries and shapes

**Key Findings:**

- Zebras remain distinctive even in grayscale due to unique stripe patterns
- Edge detection clearly separates elephants (trunk/ears) and zebras (stripes) from other species
- Buffalo and rhinos become increasingly similar without color information—foreshadowing classification challenges
- Median filtering outperforms Gaussian blur on wildlife images with complex textures

### Part B: Deep Learning Classification

The second phase implements automated species classification:

1. **Dataset Preparation**
   - Create balanced training subset (150 images/class)
   - Resize to 224×224 pixels (MobileNetV2 input size)
   - Normalize to [0,1] range

2. **Transfer Learning Model**
   - Base: MobileNetV2 pretrained on ImageNet (2.26M parameters, frozen)
   - Custom head: Global Average Pooling → Dense(128) + ReLU → Dropout(50%) → Dense(4) + Softmax
   - Trainable parameters: Only 5,124 (0.23% of total)
   - Model size: 8.63 MB (suitable for mobile deployment)

3. **Training**
   - Optimizer: Adam
   - Loss: Categorical cross-entropy
   - Epochs: 10
   - Batch size: 32

4. **Evaluation**
   - Confusion matrix analysis
   - Per-class performance metrics
   - Misclassification analysis

**Key Findings:**

- Perfect classification (100%) for highly distinctive species (elephant, zebra)
- 90% accuracy on rhino despite morphological similarity with buffalo
- Primary error (80%): Rhino → Buffalo confusion due to similar body structure, coloration, and missing horn visibility
- Transfer learning proves highly effective—achieves 95.83% accuracy training only 0.23% of parameters

## Analysis Highlights

### Visual Distinctiveness Determines Performance

- **Elephant:** Unique trunk, large ears, massive body → 100% accuracy
- **Zebra:** Distinctive black-white stripes → 100% accuracy
- **Buffalo:** Similar to rhino when horn isn't visible → 96.67% accuracy
- **Rhino:** 90% accuracy, primary challenge is rhino→buffalo confusion

### Transfer Learning Effectiveness

Despite limited training data (150 images/class), transfer learning achieved 95.83% accuracy by leveraging ImageNet features (1.4M images). The approach demonstrates that fundamental visual features (edges, textures, shapes) learned on everyday objects transfer effectively to wildlife classification.

### Interpretability Through Classical CV

Classical computer vision analysis (edge detection, contours) provided crucial insights into why deep learning succeeds or fails, enabling systematic debugging and improvement planning.

## Practical Applications

This technology enables:

- **Automated Camera Trap Analysis** - Process millions of wildlife monitoring images with 95%+ accuracy
- **Rapid Species Surveys** - Field workers can quickly identify species using mobile apps
- **Anti-Poaching Systems** - Automated rhino detection (90% accuracy) can trigger protection alerts
- **Ecological Research** - Large-scale automated classification for population and habitat analysis
- **Conservation Education** - Mobile apps to engage tourists and collect wildlife sighting data

## Understanding the Confusion Matrix

The model's primary error (rhino→buffalo) reveals important insights:

```txt
Rhino → Buffalo (4 cases, 80% of errors):
- Both share similar heavy body structure
- Similar dark gray-brown coloration
- Rhino horns not always visible in images
- Both inhabit similar savanna habitats
```

**Improvements to reduce this error:**

- Collect training images emphasizing horn visibility
- Implement attention mechanisms for head regions
- Use targeted data augmentation for problematic angles
- Fine-tune deeper model layers for more task-specific learning

## Reports and Documentation

### BRIEF_REPORT.md

4-page executive summary focusing on:

- Results summary with performance metrics
- Key challenges and solutions
- Critical interpretations and insights
- Recommendations for deployment and improvement

### PROJECT_REPORT.md

Comprehensive 10+ page detailed report including:

- Extensive classical CV analysis
- Deep learning architecture details
- Detailed per-class performance analysis
- In-depth challenge discussion
- Future research directions and recommendations

## Technical Requirements

- **TensorFlow/Keras:** For transfer learning and model training
- **OpenCV:** For classical image processing
- **NumPy:** For numerical operations
- **Matplotlib:** For visualization
- **Scikit-learn:** For confusion matrix and classification metrics
- **Pandas:** For data manipulation

### GPU Acceleration (Optional)

For faster training, install CUDA-enabled TensorFlow:

```bash
pip install tensorflow-gpu>=2.10.0
```

## 📖 How to Use

1. **Clone the repository** (see Quick Start above)
2. **Install dependencies** using pip
3. **Open the notebook** with Jupyter
4. **Run cells sequentially** to:
   - Explore classical CV techniques
   - Visualize preprocessing effects
   - Train the deep learning model
   - Analyze results and performance

Each cell includes detailed comments explaining the code and results.

## Learning Outcomes

This project demonstrates:

- Classical computer vision techniques (filtering, edge detection, contour analysis)
- Transfer learning and fine-tuning for specialized tasks
- Balancing model performance with computational efficiency
- Systematic error analysis and interpretability
- Practical wildlife monitoring applications
- Real-world constraints in computer vision deployment

## Key Insights

1. **Transfer Learning Power:** Achieves 95.83% accuracy training only 0.23% of model parameters, demonstrating the effectiveness of leveraging pretrained models for specialized tasks

2. **Visual Distinctiveness Matters:** Species with unique, easily identifiable features (stripes, trunks) achieve perfect classification, while morphologically similar species present greater challenges

3. **Classical + Modern Integration:** Combining classical computer vision interpretability with deep learning performance creates more robust, trustworthy systems

4. **Deployment Considerations:** Lightweight models (8.63 MB) enable real-world deployment on mobile and embedded systems for field conservation work

## 🔮 Future Improvements

- Implement data augmentation (rotation, flip, zoom, brightness) to improve robustness
- Unfreeze and fine-tune deeper model layers for more task-specific learning
- Collect images emphasizing distinguishing features (horn visibility for rhinos)
- Implement Grad-CAM visualization to understand model decision-making
- Test alternative architectures (ResNet50, EfficientNet, Vision Transformers)
- Deploy on camera trap networks and validate with real-world field data
- Expand to classify more African wildlife species
- Implement human-in-the-loop systems for high-stakes conservation decisions

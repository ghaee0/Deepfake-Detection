# DeepFake Video Detection Using Hybrid CNN-MLP Model

A hybrid deep learning approach for detecting manipulated facial videos using Convolutional Neural Networks (CNN) and Multi-Layer Perceptron (MLP) with facial landmark analysis.

## 📊 Results & Achievements

**High Accuracy Across Multiple Fake Video Types:**
- **Face2Face**: 98% accuracy
- **NeuralTextures**: 95% accuracy  
- **DeepFakes**: 92% accuracy
- **FaceSwap**: 90% accuracy
- **All Types Combined**: 92% accuracy

## 🏗️ Architecture

### Hybrid Model Design
Our approach combines two parallel neural networks:

1. **CNN** - Processes raw image frames for automatic feature extraction
2. **MLP** - Processes facial landmarks (eyes, lips, nose movements)  
3. **Combined** - Merges both analyses for final detection

### Facial Landmark Features
- Eye blinking patterns (EAR - Eye Aspect Ratio)
- Left/Right eye shape and movement
- Outer/Inner lip coordinates  
- Base/Top nose landmarks
- Real-time feature extraction using Dlib

## 📊 Dataset

We used the **FaceForensics++** dataset containing:
- **1,000 original video sequences**
- **4 manipulation methods**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **200 videos per manipulation type**
- **47,625 extracted frames** total

## 🔬 Key Features

- **Hybrid Architecture**: Combines CNN for visual features and MLP for structured landmark data
- **Facial Landmark Analysis**: Uses 68-point facial landmarks for biological signal detection
- **Multi-Model Ensemble**: Leverages specialized models for different manipulation types
- **Compression Robustness**: Tested on both RAW and compressed (C23) video formats
- **Real-time Capable**: Optimized for practical deployment scenarios

## 🛠️ Technical Setup

```bash
# Install required packages
pip install tensorflow opencv-python dlib numpy

# Run detection
python detect.py --video input_video.mp4

# Pins Face Classification using Neural Networks

## Project Objective
This project implements a multi-class classification system using Artificial Neural Networks (ANN) to identify and classify faces from the Pins Face Dataset.

## Dataset Description
- The dataset consists of facial images of different individuals
- Images are organized in directories, with each directory representing a different person
- Original images are of varying sizes and are preprocessed to 100x100 pixels
- Dataset is split into training (80%) and validation (20%) sets

## Instructions for Running the Code

### Setup
1. Clone the repository
```bash
git clone (https://github.com/OmarOssama25/MulticlassClassification_Pins_Face)
cd pins-face-classification
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download OpenCV face detection models
- Download `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel`
- Place them in the project root directory

### Running the Project
1. Prepare your dataset:
   - Organize images in folders, each folder named after the person
   - Place all folders in a 'dataset' directory

2. Run the training script:
```bash
python train.py
```

3. For predictions on new images:
```bash
python predict.py --image_path path/to/image
```

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn

## Installation
```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## Model Architecture
- Input Layer: Flattened 100x100x3 RGB images
- Hidden Layers:
  - Dense layer (512 neurons) with ReLU activation
  - Dense layer (256 neurons) with ReLU activation
  - Dense layer (128 neurons) with ReLU activation
- Output Layer: Dense layer with softmax activation
- Regularization: Dropout

## Expected Results
- Training accuracy: >85%
- Validation accuracy: >80%
- Model should be able to:
  - Detect faces in images
  - Classify faces with reasonable accuracy
  - Handle various lighting conditions and poses

## Performance Metrics
- Classification accuracy
- Loss values
- Confusion matrix
- Precision and recall scores

## Project Structure
```
├── dataset/
│   ├── person1/
│   ├── person2/
│   └── ...
├── models/
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Features
- Image preprocessing and augmentation
- Multi-class classification using ANN
- Model evaluation and visualization
- Random image prediction functionality

## Limitations
- Performance depends on image quality
- Requires clear facial images
- May struggle with extreme angles or poor lighting
- Limited by the size and variety of the training dataset

## Future Improvements
- Implement CNN architecture
- Add data augmentation techniques
- Implement cross-validation
- Add support for real-time classification
- Improve face detection accuracy

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Keras team for the deep learning framework
- Contributors to the Pins Face Dataset

## Contact
For any queries or suggestions, please open an issue in the repository.

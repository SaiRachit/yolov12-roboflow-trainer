# YOLOv12 Roboflow Trainer

**Complete end-to-end pipeline for training custom YOLOv12 object detection models with Roboflow datasets and testing with OpenCV.**

Perfect for beginners and professionals who want to train their own object detection models without the hassle!

## Features

- **Easy Setup** - Simple configuration with environment variables
- **Roboflow Integration** - Seamlessly download and use your Roboflow datasets
- **YOLOv12 Training** - Train state-of-the-art object detection models
- **OpenCV Testing** - Real-time detection with webcam or video files
- **Automatic Visualization** - Training graphs, confusion matrices, and predictions
- **GitHub Safe** - Secure credential management with .env files

## Prerequisites

- Python 3.8 or higher
- Webcam (optional, for real-time testing)
- GPU recommended (CUDA compatible) but works on CPU
- [Roboflow account](https://roboflow.com/) with a dataset

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yolov12-roboflow-trainer.git
cd yolov12-roboflow-trainer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics roboflow python-dotenv opencv-python
```

### 3. Configure Your Settings

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your Roboflow credentials:
```env
ROBOFLOW_API_KEY=your_actual_api_key_here
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=your_project_name
ROBOFLOW_VERSION=1

# Training settings (optional)
EPOCHS=75
IMAGE_SIZE=640
```

**Where to find these values:**
- Go to your [Roboflow project](https://app.roboflow.com/)
- Click "Download Dataset" → "Show Download Code"
- Copy the values from the code snippet

### 4. Train Your Model

```bash
python yolov12_training.py
```

Follow the interactive prompts:
- Skip the inference test (type `n`)
- Download dataset from Roboflow (type `y`)
- Start training (type `y`)

### 5. Test with OpenCV

After training completes, test your model:

```bash
python test_opencv.py
```

Choose your testing mode:
- **Webcam** - Real-time detection
- **Video file** - Test on existing videos
- **Image** - Single image detection

## Project Structure

```
yolov12-roboflow-trainer/
├── yolov12_training.py      # Main training script
├── test_opencv.py           # OpenCV testing script
├── .env                     # Your configuration (NOT in git)
├── .env.example             # Example configuration template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── datasets/                # Downloaded datasets (auto-created)
└── runs/                    # Training outputs (auto-created)
    └── detect/
        └── train/
            └── weights/
                ├── best.pt  # Your trained model (recommended)
                └── last.pt  # Last epoch model
```

## Usage Examples

### Training Your Model

```python
# The training script handles everything automatically
python yolov12_training.py

# Or customize training parameters in .env:
# EPOCHS=100
# IMAGE_SIZE=1280
```

### Using Your Trained Model in Code

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run predictions
results = model.predict('path/to/image.jpg', conf=0.5)

# Display results
results[0].show()

# Save results
results[0].save('output.jpg')
```

### Real-time Detection with OpenCV

```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    
    cv2.imshow('YOLOv12 Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Training Outputs

After training, you'll find these files in `runs/detect/train/`:

| File | Description |
|------|-------------|
| `weights/best.pt` | Best model (use this for deployment) |
| `weights/last.pt` | Final epoch model |
| `results.png` | Training metrics (loss, mAP, precision, recall) |
| `confusion_matrix.png` | Model performance visualization |
| `labels.jpg` | Label distribution in your dataset |
| `val_batch0_pred.jpg` | Sample validation predictions |

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 75 | Number of training iterations |
| `IMAGE_SIZE` | 640 | Input image size (320/640/1280) |
| `BATCH_SIZE` | Auto | Batch size (auto-calculated) |

### When to adjust:

- **More epochs (100-200)**: Better accuracy, longer training
- **Fewer epochs (25-50)**: Quick testing, lower accuracy
- **Larger image size (1280)**: Better small object detection, slower
- **Smaller image size (320)**: Faster training/inference, lower accuracy

## Troubleshooting

### API Key Issues
```
ERROR: ROBOFLOW_API_KEY not set!
```
**Solution**: Make sure `.env` file exists and contains your API key

### Dataset Download Fails
```
Error downloading dataset
```
**Solutions**:
- Verify workspace, project, and version names are correct
- Check internet connection
- Confirm you have access to the Roboflow project

### Out of Memory During Training
```
CUDA out of memory
```
**Solutions**:
- Reduce `IMAGE_SIZE` to 320 or 416
- Reduce batch size: add `batch=8` to training command
- Close other applications using GPU
- Use CPU training (slower): add `device=cpu`

### OpenCV Webcam Not Working
```
Cannot open camera
```
**Solutions**:
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)` or `(2)`
- Test camera with other apps first

## Performance Tips

### For Better Accuracy:
- Use more training data (100+ images per class)
- Balance your classes (similar number of examples)
- Increase epochs to 100-150
- Use larger image size (1280)
- Enable data augmentation in Roboflow

### For Faster Training:
- Use smaller image size (320-416)
- Reduce epochs to 50
- Use GPU if available
- Reduce batch size if out of memory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) - YOLOv12 implementation
- [Roboflow](https://roboflow.com/) - Dataset management and annotation
- [OpenCV](https://opencv.org/) - Computer vision library

## Support

Having issues? Please [open an issue](https://github.com/yourusername/yolov12-roboflow-trainer/issues) on GitHub.

## Star History

If this project helped you, please consider giving it a star!

---

**Made with care for the computer vision community**

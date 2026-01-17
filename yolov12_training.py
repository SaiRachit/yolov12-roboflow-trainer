import os
import subprocess
import sys
from pathlib import Path

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded configuration from .env file")
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables or defaults.")
    print("Install with: pip install python-dotenv")

ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
ROBOFLOW_WORKSPACE = os.getenv('ROBOFLOW_WORKSPACE', 'freedomtech')
ROBOFLOW_PROJECT = os.getenv('ROBOFLOW_PROJECT', 'yolo12-j6avk')
ROBOFLOW_VERSION = int(os.getenv('ROBOFLOW_VERSION', '2'))
DATASET_FORMAT = os.getenv('DATASET_FORMAT', 'yolov12')

# Training Configuration
EPOCHS = int(os.getenv('EPOCHS', '75'))
IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', '640'))

# Set working directory
HOME = os.getcwd()
print(f"Working directory: {HOME}")

# Install required packages
def install_packages():
    """Install required packages"""
    print("Installing required packages...")
    packages = ['ultralytics', 'roboflow', 'python-dotenv']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print("Installation complete!")

# Check ultralytics installation
def check_ultralytics():
    """Check if ultralytics is properly installed"""
    import ultralytics
    ultralytics.checks()

# Run inference with pre-trained model
def run_inference():
    """Run inference on a sample image"""
    print("\n--- Running Inference ---")
    cmd = [
        'yolo',
        'task=detect',
        'mode=predict',
        'model=yolo12s.pt',
        'conf=0.25',
        'source=https://media.roboflow.com/notebooks/examples/dog.jpeg',
        'save=True'
    ]
    subprocess.run(cmd)
    print("Inference complete! Check runs/detect/predict for results.")

# Download dataset from Roboflow
def download_dataset():
    """Download custom dataset from Roboflow"""
    print("\n--- Downloading Dataset ---")
    
    # Check if API key is set
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: ROBOFLOW_API_KEY not set!")
        print("\nTo set up your API key:")
        print("1. Create a .env file in your project directory")
        print("2. Add: ROBOFLOW_API_KEY=your_actual_key_here")
        print("3. (Optional) Also set ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION")
        print("\nOr set as environment variable:")
        print("  Windows: set ROBOFLOW_API_KEY=your_key")
        print("  Mac/Linux: export ROBOFLOW_API_KEY=your_key")
        return None
    
    # Create datasets directory
    datasets_dir = os.path.join(HOME, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    os.chdir(datasets_dir)
    
    try:
        from roboflow import Roboflow
        
        print(f"Connecting to Roboflow workspace: {ROBOFLOW_WORKSPACE}")
        print(f"Project: {ROBOFLOW_PROJECT}")
        print(f"Version: {ROBOFLOW_VERSION}")
        
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)
        dataset = version.download(DATASET_FORMAT)
        
        os.chdir(HOME)
        print(f"Dataset downloaded successfully to: {dataset.location}")
        return dataset.location
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Verify workspace, project, and version names")
        print("3. Ensure you have access to the project")
        os.chdir(HOME)
        return None

# Train YOLOv12 model
def train_model(data_path):
    """Train YOLOv12 on custom dataset"""
    print("\n--- Starting Training ---")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    
    cmd = [
        'yolo',
        'task=detect',
        'mode=train',
        'model=yolo12s.pt',
        f'data={data_path}/data.yaml',
        f'epochs={EPOCHS}',
        f'imgsz={IMAGE_SIZE}',
        'plots=True'
    ]
    subprocess.run(cmd)
    print("Training complete! Check runs/detect/train for results.")

# Main execution
def main():
    """Main execution function"""
    print("=" * 60)
    print("YOLOv12 Object Detection Training Pipeline")
    print("=" * 60)
    
    # Step 1: Install packages
    install_packages()
    
    # Step 2: Check installation
    check_ultralytics()
    
    # Step 3: Run inference (optional - comment out if not needed)
    user_input = input("\nRun inference test? (y/n): ").lower()
    if user_input == 'y':
        run_inference()
    
    # Step 4: Download dataset
    user_input = input("\nDownload dataset from Roboflow? (y/n): ").lower()
    if user_input == 'y':
        data_path = download_dataset()
        if data_path is None:
            print("\nDataset download failed. Exiting...")
            return
    else:
        data_path = input("Enter path to your dataset directory: ")
        if not os.path.exists(data_path):
            print(f"Error: Path '{data_path}' does not exist!")
            return
    
    # Step 5: Train model
    user_input = input("\nStart training? (y/n): ").lower()
    if user_input == 'y':
        train_model(data_path)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
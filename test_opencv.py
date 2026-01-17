import cv2
from ultralytics import YOLO
import os
import argparse
from pathlib import Path


class ObjectDetector:
    
    def __init__(self, model_path='runs/detect/train/weights/best.pt', conf=0.5, iou=0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        print(f"Model loaded successfully!")
        print(f"Classes: {self.model.names}")
    
    def detect_objects(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy()
        scores = detections.boxes.conf.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            score = scores[i]
            class_id = int(classes[i])
            label = f"{self.model.names[class_id]}: {score:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def detect_webcam(self, camera_id=0, width=640, height=480):
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open camera {camera_id}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("\n=== Webcam Detection ===")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                annotated_frame = self.detect_objects(frame)
                
                frame_count += 1
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("YOLOv12 Webcam Detection", annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    output_path = f"detection_frame_{frame_count}.jpg"
                    cv2.imwrite(output_path, annotated_frame)
                    print(f"Saved frame to {output_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nTotal frames processed: {frame_count}")
    
    def detect_video(self, video_path, output_path=None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n=== Video Detection ===")
        print(f"Input: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output: {output_path}")
        
        print("\nPress 'q' to quit early")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                annotated_frame = self.detect_objects(frame)
                
                progress = (frame_count / total_frames) * 100
                cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if writer:
                    writer.write(annotated_frame)
                
                cv2.imshow("YOLOv12 Video Detection", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
                
                if frame_count % 30 == 0:
                    print(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count}/{total_frames} frames")
    
    def detect_image(self, image_path, output_path=None, show=True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n=== Image Detection ===")
        print(f"Input: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Error: Could not read image {image_path}")
        
        annotated_frame = self.detect_objects(frame)
        
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Output saved to: {output_path}")
        
        if show:
            print("Press any key to close the window")
            cv2.imshow("YOLOv12 Image Detection", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated_frame


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='YOLOv12 Object Detection with OpenCV')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'video', 'image'],
                       help='Detection mode')
    parser.add_argument('--source', type=str, default='0',
                       help='Source: camera ID for webcam, file path for video/image')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for video/image')
    
    args = parser.parse_args()
    
    detector = ObjectDetector(args.model, args.conf, args.iou)
    
    if args.mode == 'webcam':
        camera_id = int(args.source) if args.source.isdigit() else 0
        detector.detect_webcam(camera_id)
    
    elif args.mode == 'video':
        output = args.output if args.output else f"output_{Path(args.source).name}"
        detector.detect_video(args.source, output)
    
    elif args.mode == 'image':
        output = args.output if args.output else f"output_{Path(args.source).name}"
        detector.detect_image(args.source, output)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("=" * 60)
        print("YOLOv12 Object Detection - Interactive Mode")
        print("=" * 60)
        
        default_model = 'runs/detect/train/weights/best.pt'
        if not os.path.exists(default_model):
            print(f"\nWarning: Default model not found at {default_model}")
            model_path = input("Enter path to your model (.pt file): ").strip()
        else:
            model_path = default_model
        
        print("\nSelect detection mode:")
        print("1. Webcam (real-time)")
        print("2. Video file")
        print("3. Image file")
        mode_choice = input("Enter choice (1-3): ").strip()
        
        try:
            detector = ObjectDetector(model_path)
            
            if mode_choice == '1':
                detector.detect_webcam()
            
            elif mode_choice == '2':
                video_path = input("Enter path to video file: ").strip()
                save_output = input("Save output video? (y/n): ").strip().lower()
                output_path = f"output_{Path(video_path).name}" if save_output == 'y' else None
                detector.detect_video(video_path, output_path)
            
            elif mode_choice == '3':
                image_path = input("Enter path to image file: ").strip()
                save_output = input("Save output image? (y/n): ").strip().lower()
                output_path = f"output_{Path(image_path).name}" if save_output == 'y' else None
                detector.detect_image(image_path, output_path)
            
            else:
                print("Invalid choice!")
        
        except Exception as e:
            print(f"\nError: {e}")
    
    else:
        main()
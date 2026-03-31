import cv2
import os
from ultralytics import YOLO
import torch

class ProductDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """
        Initialize YOLOv8 model for product detection
        
        Args:
            model_name: YOLOv8 model size (nano, small, medium, large, xlarge)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.output_dir = 'products'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect(self, image_path, confidence=0.5):
        """
        Detect products in an image
        
        Args:
            image_path: Path to image file
            confidence: Confidence threshold for detections
        
        Returns:
            Detection results
        """
        results = self.model(image_path, conf=confidence)
        return results
    
    def detect_batch(self, image_paths, confidence=0.5):
        """
        Detect products in multiple images
        
        Args:
            image_paths: List of image file paths
            confidence: Confidence threshold
        
        Returns:
            List of detection results
        """
        all_results = []
        for image_path in image_paths:
            results = self.detect(image_path, confidence)
            all_results.append({
                'image': image_path,
                'results': results
            })
        return all_results
    
    def save_detections(self, results, image_path):
        """Save detected products with bounding boxes"""
        if results:
            output_path = os.path.join(
                self.output_dir, 
                os.path.basename(image_path)
            )
            results[0].save(output_path)
            return output_path
        return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        detector = ProductDetector()
        results = detector.detect(sys.argv[1])
        print(f"Found {len(results[0].boxes)} products")
import os
import json
import csv
import argparse
from extract_frames import FrameExtractor
from detect_products import ProductDetector
from extract_text import TextExtractor

class SupermarketVisionPipeline:
    def __init__(self, video_path, frame_interval=30, confidence=0.5):
        """
        Initialize the complete pipeline
        
        Args:
            video_path: Path to input video file
            frame_interval: Extract every nth frame
            confidence: Confidence threshold for detections
        """
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.confidence = confidence
        
        # Initialize components
        self.frame_extractor = FrameExtractor()
        self.product_detector = ProductDetector()
        self.text_extractor = TextExtractor()
        
        self.dataset = []
    
    def run(self):
        """Execute the complete pipeline"""
        print("=" * 50)
        print("🚀 SUPERMARKET VISION PIPELINE")
        print("=" * 50)
        
        # Step 1: Extract frames
        print("\n[1/3] Extracting frames from video...")
        frames = self.frame_extractor.extract(self.video_path, self.frame_interval)
        print(f"✓ Extracted {len(frames)} frames")
        
        # Step 2: Detect products
        print("\n[2/3] Detecting products in frames...")
        detections = self.product_detector.detect_batch(frames, self.confidence)
        print(f"✓ Processed {len(detections)} frames")
        
        # Step 3: Extract text (product names, prices)
        print("\n[3/3] Extracting text from images...")
        for i, detection in enumerate(detections):
            image_path = detection['image']
            results = detection['results']
            
            text_data = self.text_extractor.extract_text(image_path)
            
            entry = {
                'frame': i,
                'image': image_path,
                'products_detected': len(results[0].boxes) if results else 0,
                'text_extracted': text_data
            }
            self.dataset.append(entry)
        
        print(f"✓ Extracted text from {len(self.dataset)} frames")
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
    
    def save_results(self):
        """Save results to JSON and CSV"""
        # Save as JSON
        json_path = 'dataset.json'
        with open(json_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"\n📄 Saved JSON: {json_path}")
        
        # Save as CSV
        csv_path = 'dataset.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Image Path', 'Products Detected', 'Text Extracted'])
            
            for entry in self.dataset:
                text_list = ', '.join([item['text'] for item in entry['text_extracted']])
                writer.writerow([
                    entry['frame'],
                    entry['image'],
                    entry['products_detected'],
                    text_list
                ])
        print(f"📊 Saved CSV: {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supermarket Vision Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--interval', type=int, default=30, help='Frame extraction interval')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    pipeline = SupermarketVisionPipeline(
        args.video,
        frame_interval=args.interval,
        confidence=args.confidence
    )
    pipeline.run()
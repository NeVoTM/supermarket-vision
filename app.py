import streamlit as st
import os
import json
import csv
from pathlib import Path
from extract_frames import FrameExtractor
from detect_products import ProductDetector
from extract_text import TextExtractor
import cv2
import numpy as np

st.set_page_config(page_title="Supermarket Vision", layout="wide")

st.markdown("# 🛒 Supermarket Vision Pipeline")
st.markdown("### Upload a video to analyze products, detect items, and extract text")

# Create temp directories
os.makedirs('temp_video', exist_ok=True)
os.makedirs('temp_results', exist_ok=True)

# Sidebar configuration
st.sidebar.markdown("## ⚙️ Configuration")
frame_interval = st.sidebar.slider("Frame Extraction Interval", 1, 100, 30)
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
ocr_confidence = st.sidebar.slider("OCR Confidence", 0.1, 1.0, 0.3)

# File upload
uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded video
    video_path = os.path.join('temp_video', uploaded_file.name)
    with open(video_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Video uploaded: {uploaded_file.name}")
    
    # Display video info
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("FPS", fps)
    with col2:
        st.metric("Total Frames", frame_count)
    with col3:
        st.metric("Duration (s)", f"{duration:.2f}")
    
    # Process button
    if st.button("🚀 Start Analysis", key="process"):
        with st.spinner("Processing video..."):
            try:
                # Step 1: Extract frames
                st.info("📹 Step 1: Extracting frames...")
                frame_extractor = FrameExtractor(output_dir='temp_results/frames')
                frames = frame_extractor.extract(video_path, frame_interval)
                st.success(f"✅ Extracted {len(frames)} frames")
                
                # Step 2: Detect products
                st.info("🎯 Step 2: Detecting products...")
                product_detector = ProductDetector()
                detections = product_detector.detect_batch(frames, confidence_threshold)
                st.success(f"✅ Processed {len(detections)} frames")
                
                # Step 3: Extract text
                st.info("📝 Step 3: Extracting text (OCR)...")
                text_extractor = TextExtractor()
                dataset = []
                
                for i, detection in enumerate(detections):
                    image_path = detection['image']
                    results = detection['results']
                    
                    text_data = text_extractor.extract_text(image_path, ocr_confidence)
                    
                    entry = {
                        'frame_id': i,
                        'image_path': image_path,
                        'products_detected': len(results[0].boxes) if results and len(results) > 0 else 0,
                        'text_extracted': text_data
                    }
                    dataset.append(entry)
                
                st.success(f"✅ Text extracted from {len(dataset)} frames")
                
                # Save results
                st.info("💾 Saving results...")
                json_path = 'temp_results/dataset.json'
                csv_path = 'temp_results/dataset.csv'
                
                with open(json_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frame ID', 'Products Detected', 'Text Extracted', 'Confidence Scores'])
                    
                    for entry in dataset:
                        text_list = '; '.join([f"{item['text']} ({item['confidence']:.2f})" 
                                              for item in entry['text_extracted']])
                        writer.writerow([
                            entry['frame_id'],
                            entry['products_detected'],
                            text_list if text_list else 'None'
                        ])
                
                st.success("✅ Results saved!")
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                total_products = sum(e['products_detected'] for e in dataset)
                total_text_items = sum(len(e['text_extracted']) for e in dataset)
                avg_products = total_products / len(dataset) if dataset else 0
                
                with col1:
                    st.metric("Total Products Detected", total_products)
                with col2:
                    st.metric("Avg Products per Frame", f"{avg_products:.1f}")
                with col3:
                    st.metric("Text Items Extracted", total_text_items)
                
                # Display sample results
                st.markdown("### Sample Results")
                tabs = st.tabs(["Frames with Detections", "Text Extraction", "Raw Data"])
                
                with tabs[0]:
                    st.markdown("#### Detected Products")
                    for i, detection in enumerate(detections[:3]):  # Show first 3
                        image_path = detection['image']
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"Frame {i} - {len(detection['results'][0].boxes) if detection['results'] else 0} products")
                
                with tabs[1]:
                    st.markdown("#### Extracted Text")
                    for i, entry in enumerate(dataset[:5]):  # Show first 5
                        if entry['text_extracted']:
                            st.markdown(f"**Frame {entry['frame_id']}:**")
                            for text_item in entry['text_extracted']:
                                st.write(f"- {text_item['text']} (confidence: {text_item['confidence']:.2%})")
                        else:
                            st.write(f"Frame {entry['frame_id']}: No text detected")
                
                with tabs[2]:
                    st.markdown("#### Raw JSON Data")
                    st.json(dataset[:2])  # Show first 2 entries
                
                # Download buttons
                st.markdown("---")
                st.markdown("## 📥 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(json_path, 'r') as f:
                        st.download_button(
                            label="📄 Download JSON",
                            data=f.read(),
                            file_name="dataset.json",
                            mime="application/json"
                        )
                
                with col2:
                    with open(csv_path, 'r') as f:
                        st.download_button(
                            label="📊 Download CSV",
                            data=f.read(),
                            file_name="dataset.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.error("Please check your video file and try again.")

else:
    st.info("👆 Upload a video file to get started!")
    st.markdown("""
    ### How it works:
    1. **Upload** a supermarket walkthrough video
    2. **Configure** extraction parameters (frame interval, confidence thresholds)
    3. **Click** "Start Analysis"
    4. **View** results with detected products and extracted text
    5. **Download** dataset as JSON or CSV
    """
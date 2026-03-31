# Supermarket Computer Vision Pipeline (Open Source)

This project processes a supermarket walkthrough video and extracts information about products, price tags, product names, and produces structured data.

All tools used are **free and open source**.

---

# 🚀 What This Project Does

1. Extract frames from a supermarket video.
2. Detect products in each frame using YOLOv8 (open source).
3. Detect and crop price tags.
4. Use EasyOCR (free OCR) to read:
   - product names
   - price tags
5. Save all data to:
   - `dataset.json`
   - `dataset.csv`

---

# 🧰 Tools Used (ALL FREE)

- Python
- OpenCV
- Ultralytics YOLOv8 (free model)
- EasyOCR
- NumPy

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

---

# 🎬 Usage

```bash
python pipeline.py --video path/to/video.mp4
```

---

# 📁 Project Structure

```
supermarket-vision/
├── README.md
├── requirements.txt
├── extract_frames.py
├── detect_products.py
├── extract_text.py
├── pipeline.py
├── frames/
├── products/
└── prices/
```

---

# 📊 Output

- `dataset.json` - Structured data with products and prices
- `dataset.csv` - CSV format for easy analysis

---

# 📝 License

MIT License - Free to use and modify
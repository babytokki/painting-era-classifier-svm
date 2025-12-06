# Painting Era Classifier 𐔌՞. .՞𐦯
Identifying artistic periods using classical computer vision.

# Overview
A web-based image classifier that identifies whether a painting belongs to the **Baroque**, **Medieval**, or **Renaissance** era using **classical computer vision methods** (HOG features + SVM).  
The model is deployed using **Streamlit**. Users can upload a painting image and instantly see:
- Predicted painting era
- Confidence score
- Short description
- Gold-themed UI
- Confidence bar chart (Altair)

# Features
- HOG features + SVM
- Custom styled UI
- Live prediction and confidence chart
- Supports JPG / JPEG / PNG
- Runs locally via Streamlit

# Model
Raw Dataset: Painting Eras Detection Classification Dataset by ArtAncestry
https://share.google/rqdLnLG0PWmTDK0zf

Trained Dataset:
https://drive.google.com/drive/folders/10H5t042JTYi7Mli0XM-xdCBK6hQBXZvv?usp=drive_link

Trained on 3 classes:
- Baroque paintings
- Medieval art
- Renaissance paintings

Preprocessing:
- Resize to 128×128
- Convert RGB → grayscale

Feature Extraction:
Using **HOG (Histogram of Oriented Gradients)** from `skimage.feature.hog`:
- `orientations=9`  
- `pixels_per_cell=(8, 8)`  
- `cells_per_block=(2, 2)`  
- `block_norm="L2-Hys"`

Classifier:
A **Linear SVM** (`sklearn.svm.SVC`) trained with:

```python
SVC(kernel="linear", probability=True)

Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curves (One-vs-Rest)

Deployment (Streamlit UI):
- Custom gold-themed interface
- Probability bar chart (Altair)
- Short era descriptions
- Clean uploading workflow

# Installation
1️⃣ Clone the repository:
- git clone https://github.com/babytokki/painting-era-classifier.git
- cd painting-era-classifier

2️⃣ Install dependencies:
- pip install -r requirements.txt

▶️ Run the App:
- run main.ipynb
- streamlit run app.py

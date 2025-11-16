# Problem 2 — Sentiment Classification Using Transformer Embeddings + SVM

## Overview
This project implements a sentiment analysis system that classifies airline customer reviews as **positive** or **negative**.  
The solution uses a **hybrid machine learning approach** combining:

- **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) for semantic text representation  
- **Linear SVM (Support Vector Machine)** for classification  

This approach is:
- 100% **free**
- Fully **offline**
- **Fast** and **accurate**
- Suitable for production-level pipelines and placement demonstrations

The dataset used is `2026_dataset.csv`, containing 2088 rows of airline reviews.

---

## Approach

### 1. Dataset Used
The project uses two columns from the dataset:

- **Review** → Customer feedback text  
- **Recommended** → “yes/no” sentiment indicator  

The `Recommended` column is normalized:
- `"yes"` → **positive**
- `"no"`  → **negative**

### 2. Embedding Generation
Each review (text column) along with recommend (label column) is converted into a dense vector using the SentenceTransformer model:

- Model: `all-MiniLM-L6-v2`
- Embedding Dimension: **384**
- Captures contextual meaning of text
- Runs fully offline after initial download

Generated files:
```
saved/embeddings.npy
saved/labels.npy
```

### 3. Training the Classifier
A **Linear SVM** classifier is trained using the embeddings.

Why SVM?
- Strong performance on high-dimensional embeddings  
- Robust and stable  
- Efficient for small to medium-sized datasets  

Model saved at:
```
saved/embedding_svm.pkl
```

### 4. Prediction Pipeline
When a new review is entered:
1. Generate its embedding  
2. Classify using the trained SVM  
3. Output sentiment + confidence score  

Example prediction:
```
Sentiment: positive
Source: embedding-svm
Confidence: 0.89
```

---

## Folder Structure

```
problem2/
│── main.py
│── requirements.txt
│── README.md
│
├── data/
│     └── 2026_dataset.csv
│
├── model/
│     ├── embedding_classifier.py
│     ├── hybrid_predictor.py
│     └── __init__.py
│
├── utils/
│     ├── preprocessing.py
│     └── __init__.py
│
└── saved/
      (created after training)
```

---

## Installation

### Step 1 — Install Dependencies
Run inside the `problem2/` directory:

```
pip install -r requirements.txt
```

This installs:
- pandas  
- numpy  
- scikit-learn  
- sentence-transformers  
- tqdm  

---

## Running the Project

### **1. Generate Embeddings**
Converts all reviews into transformer embeddings.

```
python main.py embed
```

Outputs:
```
saved/embeddings.npy
saved/labels.npy
```

---

### **2. Train the SVM Classifier**

```
python main.py train
```

Outputs:
```
saved/embedding_svm.pkl
```

---

### **3. Run Sentiment Inference**

```
python main.py infer
```

You will be prompted:

```
Enter customer feedback:
```

Type any review text.

---

## Example

**Input:**
```
The flight was delayed but the staff were very helpful.
```

**Output:**
```
Sentiment: positive
Source: embedding-svm
Confidence: 0.92
```

---

## Notes
- This system runs entirely offline (after initial model download).  
- No API keys or external services are required.  
- Suitable for real-world ML tasks and placement demonstrations.  

---

## Summary
This project demonstrates a modern and efficient sentiment classifier using:

- Transformer-based embeddings for semantic understanding  
- A classical SVM classifier for fast and accurate predictions  
- A clean, modular, and production-ready architecture  

The solution is simple to run, highly accurate, and ready for deployment or academic evaluation.

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
import pickle
import os

# Load embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    return embedding_model.encode(text)


def create_embedding_dataset(data):
    X = []
    y = []

    print("Generating embeddings using SentenceTransformer...")
    for text, label in tqdm(data):
        emb = get_embedding(text)
        X.append(emb)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    os.makedirs("saved", exist_ok=True)
    np.save("saved/embeddings.npy", X)
    np.save("saved/labels.npy", y)

    return X, y


def train_svm_classifier(X, y):
    print("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    with open("saved/embedding_svm.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Model saved to saved/embedding_svm.pkl")
    return clf


def load_classifier():
    with open("saved/embedding_svm.pkl", "rb") as f:
        return pickle.load(f)

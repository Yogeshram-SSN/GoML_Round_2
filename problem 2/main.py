import sys
import numpy as np
from utils.preprocessing import load_and_prepare_dataset
from model.embedding_classifier import create_embedding_dataset, train_svm_classifier
from model.hybrid_predictor import hybrid_predict

DATA_PATH = "data/2026_dataset.csv"


def embed():
    print("Loading data...")
    data = load_and_prepare_dataset(DATA_PATH)
    create_embedding_dataset(data)


def train():
    print("Loading embeddings...")
    X = np.load("saved/embeddings.npy")
    y = np.load("saved/labels.npy")
    train_svm_classifier(X, y)


def infer():
    text = input("Enter customer feedback: ")
    sentiment, source, confidence = hybrid_predict(text)
    print("\n=== RESULT ===")
    print("Sentiment:", sentiment)
    print("Source:", source)
    print("Confidence:", confidence)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [embed | train | infer]")
        exit()

    cmd = sys.argv[1]

    if cmd == "embed":
        embed()
    elif cmd == "train":
        train()
    elif cmd == "infer":
        infer()
    else:
        print("Invalid command. Use embed/train/infer")

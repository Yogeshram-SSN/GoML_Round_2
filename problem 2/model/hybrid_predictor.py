import numpy as np
from .embedding_classifier import get_embedding, load_classifier

def hybrid_predict(text, threshold=0.80):
    clf = load_classifier()

    emb = get_embedding(text).reshape(1, -1)
    probas = clf.predict_proba(emb)[0]
    classes = clf.classes_

    max_idx = np.argmax(probas)
    pred = classes[max_idx]
    confidence = probas[max_idx]

    # No LLM fallback — using only SVM + embeddings
    return pred, "embedding-svm", confidence

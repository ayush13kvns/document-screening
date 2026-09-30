"""Portable JSON model: no pickle/joblib execution during loading."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b\w\w+\b"

def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    # Remove transport markup while retaining its text. This is NOT an HTML renderer.
    text = re.sub(r"<[^>]{0,500}>", " ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " emailtoken ", text)
    text = re.sub(r"https?://\S+", " urltoken ", text)
    return " ".join(text.split())

class Model:
    def __init__(self, path):
        path = Path(path)
        if path.stat().st_size > 25_000_000:
            raise ValueError("Model artifact exceeds 25 MB")
        raw = path.read_bytes()
        self.version = hashlib.sha256(raw).hexdigest()[:16]
        self.data = json.loads(raw)
        d = self.data
        if d.get("schema_version") != 1:
            raise ValueError("Unsupported model schema")
        vocab = d["vocabulary"]
        if not vocab or len(vocab)>50000 or sorted(vocab.values())!=list(range(len(vocab))):
            raise ValueError("Invalid vocabulary")
        self.vectorizer = TfidfVectorizer(vocabulary=vocab, ngram_range=(1,2),
                                         sublinear_tf=True, token_pattern=TOKEN_PATTERN)
        self.vectorizer.idf_ = np.asarray(d["idf"], dtype=float)
        self.coef = np.asarray(d["coef"], dtype=float)
        scalars = [d["intercept"], d["calibration_coef"], d["calibration_intercept"]]
        if self.coef.shape != (len(vocab),) or self.vectorizer.idf_.shape != self.coef.shape:
            raise ValueError("Model shape mismatch")
        if not all(np.isfinite(x).all() for x in [self.coef,self.vectorizer.idf_,np.array(scalars)]):
            raise ValueError("Model contains non-finite values")
        self.demo = bool(d["demo"])

    def score(self, texts):
        normalized = [normalize(t) for t in texts]
        x = self.vectorizer.transform(normalized)
        logits = np.asarray(x @ self.coef).ravel() + self.data["intercept"]
        probabilities = expit(logits*self.data["calibration_coef"] + self.data["calibration_intercept"])
        coverage = []
        for text in normalized:
            tokens = re.findall(TOKEN_PATTERN, text)
            coverage.append(sum(t in self.data["vocabulary"] for t in tokens)/max(1,len(tokens)))
        return probabilities, coverage

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from sklearn.model_selection import train_test_split
import json

class TextClassifier:
    """A simple two‑layer neural network (input → ReLU → softmax) for text‑category classification.

    The model autosaves to ``model_path`` whenever the validation loss improves.  At runtime it
    first tries to *load* an existing model; if one is found, training is skipped entirely so you
    can jump straight to inference.
    """

    def __init__(self, hidden_size: int = 8, learning_rate: float = 0.1, model_path: str = "best_model.pkl"):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vectorizer = CountVectorizer(binary=True)
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.best_loss = float("inf")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    @staticmethod
    def _compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = y_true.shape[0]
        return -np.mean(np.log(y_pred[np.arange(m), y_true.argmax(axis=1)] + 1e-8))

    def _backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        m = y_true.shape[0]
        dZ2 = y_pred - y_true
        dW2 = (1 / m) * self.a1.T @ dZ2
        db2 = (1 / m) * dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.a1 > 0)
        dW1 = (1 / m) * X.T @ dZ1
        db1 = (1 / m) * dZ1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def _update_weights(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def _save_model(self):
        model_data = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "best_loss": self.best_loss,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {self.model_path} (val loss={self.best_loss:.4f})")

    def load_model(self) -> bool:
        """Return ``True`` if a saved model was found and loaded."""
        if not os.path.exists(self.model_path):
            return False
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        self.W1 = model_data["W1"]
        self.b1 = model_data["b1"]
        self.W2 = model_data["W2"]
        self.b2 = model_data["b2"]
        self.vectorizer = model_data["vectorizer"]
        self.label_encoder = model_data["label_encoder"]
        self.hidden_size = model_data["hidden_size"]
        self.learning_rate = model_data["learning_rate"]
        self.best_loss = model_data["best_loss"]
        print(f"Loaded model from {self.model_path} (val loss={self.best_loss:.4f})")
        return True

    def fit(self, texts, labels, epochs: int = 1000, validation_split: float = 0.02):
        """Train the network, autosaving whenever the validation loss improves."""
        X = self.vectorizer.fit_transform(texts).toarray()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = np.eye(len(self.label_encoder.classes_))[y_encoded]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        input_size = X_train.shape[1]
        output_size = len(self.label_encoder.classes_)

        # He init
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, output_size))

        # If an older model exists *but we still chose to train*, preserve its best_loss threshold
        if self.best_loss == float("inf"):
            self.best_loss = 1e9

        for epoch in range(epochs):
            y_pred = self._forward(X_train)
            loss = self._compute_loss(y_train, y_pred)
            dW1, db1, dW2, db2 = self._backward(X_train, y_train, y_pred)
            self._update_weights(dW1, db1, dW2, db2)

            # Validation
            val_pred = self._forward(X_val)
            val_loss = self._compute_loss(y_val, val_pred)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:5d} │ train={loss:.4f} │ val={val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_model()

    def predict(self, text: str) -> str:
        X = self.vectorizer.transform([text]).toarray()
        probabilities = self._forward(X)[0]
        idx = int(np.argmax(probabilities))
        predicted_class = self.label_encoder.classes_[idx]

        print("\nProbabilities:")
        for cls, prob in zip(self.label_encoder.classes_, probabilities):
            print(f"{cls:>24}: {prob * 100:6.2f}%")
        return predicted_class

def load_dataset(json_path: str = "sample_data.json"):
    """Return ``texts`` and ``labels`` lists extracted from ``sample_data.json``."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data["categories"]
    groups = [
        ("greeting", categories["greetings"]),
        ("farewell", categories["farewells"]),
        ("question", categories["questions"]),
        ("appreciation", categories["appreciation"]),
        ("apology", categories["apologies"]),
        ("compliment", categories["compliments"]),
        ("agreement", categories["agreements"]),
        ("disagreement", categories["disagreements"]),
        ("request", categories["requests"]),
        ("emotional_expression", categories["emotional_expressions"]),
    ]

    texts, labels = [], []
    for label, samples in groups:
        texts.extend(samples)
        labels.extend([label] * len(samples))
    return texts, labels


def main():
    texts, labels = load_dataset()

    classifier = TextClassifier(hidden_size=400, model_path="best_text_classifier.pkl")

    if classifier.load_model():
        print("Model loaded — skipping training.\n")
    else:
        print("No saved model found — starting training…\n")
        classifier.fit(texts, labels, epochs=10000)
        print("\nTraining complete.\n")

    print("Available categories:", list(classifier.label_encoder.classes_))
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "quit":
            break
        prediction = classifier.predict(user_input)
        print(f"Predicted category: {prediction}\n")


if __name__ == "__main__":
    main()
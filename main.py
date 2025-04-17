import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from sklearn.model_selection import train_test_split
import json

class TextClassifier:
    def __init__(self, hidden_size=8, learning_rate=0.1, model_path='best_model.pkl'):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vectorizer = CountVectorizer(binary=True)
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.best_loss = float('inf')

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.mean(np.log(y_pred[np.arange(m), y_true.argmax(axis=1)] + 1e-8))
        return loss

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        dZ2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.a1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def save_model(self):
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'best_loss': self.best_loss
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {self.model_path} with loss {self.best_loss:.4f}")

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.W1 = model_data['W1']
            self.b1 = model_data['b1']
            self.W2 = model_data['W2']
            self.b2 = model_data['b2']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.hidden_size = model_data['hidden_size']
            self.learning_rate = model_data['learning_rate']
            self.best_loss = model_data['best_loss']
            print(f"Loaded model with loss {self.best_loss:.4f}")
            return True
        return False

    def fit(self, texts, labels, epochs=1000, validation_split=0.002):
        # Split data into training and validation sets
        X = self.vectorizer.fit_transform(texts).toarray()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = np.eye(len(self.label_encoder.classes_))[y_encoded]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Initialize weights
        input_size = X_train.shape[1]
        output_size = len(self.label_encoder.classes_)

        # He initialization
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, output_size) * np.sqrt(2./self.hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Try to load existing model
        if self.load_model():
            # Evaluate on validation set to get current best loss
            val_pred = self.forward(X_val)
            self.best_loss = self.compute_loss(y_val, val_pred)

        for epoch in range(epochs):
            # Training
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_train, y_pred)
            dW1, db1, dW2, db2 = self.backward(X_train, y_train, y_pred)
            self.update_weights(dW1, db1, dW2, db2)

            # Validation
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save if validation loss improves
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model()

    def predict(self, text):
        X = self.vectorizer.transform([text]).toarray()
        probabilities = self.forward(X)[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_index]

        print("\nProbabilities:")
        for cls, prob in zip(self.label_encoder.classes_, probabilities):
            print(f"{cls}: {prob*100:.2f}%")

        return predicted_class

with open('sample_data.json', 'r') as f:
    data = json.load(f)

    # Load Category
    categories = data["categories"]

    # Load Sample Datas
    greetings = categories["greetings"]
    farewells = categories["farewells"]
    questions = categories["questions"]
    appreciation = categories["appreciation"]
    apologies = categories["apologies"]
    compliments = categories["compliments"]
    agreements = categories["agreements"]
    disagreements = categories["disagreements"]
    requests = categories["requests"]
    emotional_expressions = categories["emotional_expressions"]


# Combine all data
texts = greetings + farewells + questions + appreciation + apologies + compliments + agreements + disagreements + requests + emotional_expressions
labels = ["greeting"]*len(greetings) + ["farewell"]*len(farewells) + ["question"]*len(questions) + ["appreciation"]*len(appreciation) + ["apology"]*len(apologies) + ["compliment"]*len(compliments) + ["agreement"]*len(agreements) + ["disagreement"]*len(disagreements) + ["request"]*len(requests) + ["emotional_expression"]*len(emotional_expressions)

# Train classifier with autosave
classifier = TextClassifier(hidden_size=400, model_path='best_text_classifier.pkl')
classifier.fit(texts, labels, epochs=10000)

# Interactive testing
print("\nAvailable categories:", classifier.label_encoder.classes_)
while True:
    user_input = input("\n>>> ")
    if user_input.lower() == 'quit':
        break
    prediction = classifier.predict(user_input)
    print(f"Predicted category: {prediction}")

    if prediction == "greeting":
        print("Lyne: Hello! How can I assist you today?")
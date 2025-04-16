import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class TextClassifier:
    def __init__(self, hidden_size=8, learning_rate=0.1):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.vectorizer = CountVectorizer(binary=True)
        self.label_encoder = LabelEncoder()
    
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
    
    def fit(self, texts, labels, epochs=1000):
        # Convert text to numerical features
        X = self.vectorizer.fit_transform(texts).toarray()
        
        # Convert string labels to numerical indices
        self.label_encoder.fit(labels)
        y_encoded = self.label_encoder.transform(labels)
        
        # Convert to one-hot encoding
        y = np.eye(len(self.label_encoder.classes_))[y_encoded]
        
        # Initialize weights based on actual input size
        input_size = X.shape[1]
        output_size = len(self.label_encoder.classes_)
        
        # He initialization
        self.W1 = np.random.randn(input_size, self.hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, output_size) * np.sqrt(2./self.hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            self.update_weights(dW1, db1, dW2, db2)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, text):
        X = self.vectorizer.transform([text]).toarray()
        probabilities = self.forward(X)[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_index]
        
        print("Probabilities:")
        for cls, prob in zip(self.label_encoder.classes_, probabilities):
            print(f"{cls}: {prob*100:.2f}%")
        
        return predicted_class

# Sample dataset with multiple categories
greetings = ["hello", "hi there", "good morning", "hey", "howdy", "wassup", "sup", "yo"]
farewells = ["goodbye", "bye", "see you", "farewell", "take care", "good night"]
questions = ["how are you", "what's up", "how is it going", "how do you do"]
appreciation = ["thank you", "thanks", "I appreciate it", "that's kind of you"]

# Combine all data
texts = greetings + farewells + questions + appreciation
labels = ["greeting"]*len(greetings) + ["farewell"]*len(farewells) + ["question"]*len(questions) + ["appreciation"]*len(appreciation)

# Train classifier (no need to specify input size now)
classifier = TextClassifier(hidden_size=8)
classifier.fit(texts, labels, epochs=100000)

# Interactive testing
print("\nAvailable categories:", classifier.label_encoder.classes_)
while True:
    user_input = input("\nEnter a phrase (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    prediction = classifier.predict(user_input)
    print(f"Predicted category: {prediction}")
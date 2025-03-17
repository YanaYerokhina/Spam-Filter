import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Convert labels to binary (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Build a Pipeline: Vectorization + Naïve Bayes Classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text into word count vectors
    ('tfidf', TfidfTransformer()),      # Normalize word frequencies
    ('classifier', MultinomialNB())     # Apply Naïve Bayes
])

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Test with New Messages
test_messages = [
    "Free money now!!!",
    "Hey, are we meeting tomorrow?",
    "Congratulations! You won a lottery."
]
predictions = model.predict(test_messages)

for msg, pred in zip(test_messages, predictions):
    print(f'Message: {msg} --> {"Spam" if pred == 1 else "Not Spam"}')

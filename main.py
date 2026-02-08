import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

print("Dataset Preview:")
print(data.head())

# Input & Output
X = data["message"]
y = data["label"]

# Convert text → numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# User input
msg = input("\nEnter a message: ")
msg_vector = vectorizer.transform([msg])

prediction = model.predict(msg_vector)

print("Prediction:", prediction[0])

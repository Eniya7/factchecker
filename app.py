import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "text": [
        "Government launches new health scheme",
        "Aliens landed in Chennai yesterday",
        "Stock market reaches new high",
        "Drinking bleach cures COVID",
        "New education policy introduced"
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def check_news(news):
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)
    return "Real News ✅" if prediction[0] == 1 else "Fake News ❌"

# Test
user_input = input("Enter news: ")
print(check_news(user_input))
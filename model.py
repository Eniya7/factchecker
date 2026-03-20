from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
texts = [
    "India won the cricket match",
    "Government announced new education policy",
    "PM launched new infrastructure project",
    "Stock market reached new high",
    "New hospital opened in Chennai",
    "ISRO launched satellite successfully",

    "Aliens landed in Chennai",
    "Drinking petrol cures disease",
    "Earth will end tomorrow",
    "Humans can live without oxygen",
    "Eating plastic improves health",
    "Sun rises from west tomorrow"
]

labels = [1,1,1,1,1,1, 0,0,0,0,0,0]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Model
model = LogisticRegression(class_weight='balanced')
model.fit(X, labels)

# Prediction function
def predict_news(news):
    x = vectorizer.transform([news])
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]

    confidence = round(max(prob) * 100, 2)
    result = "REAL" if pred == 1 else "FAKE"

    return result, confidence
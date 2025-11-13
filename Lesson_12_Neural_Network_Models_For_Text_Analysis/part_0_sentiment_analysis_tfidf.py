from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense

# Sample data - sentiment labeled
documents = [
    "I love this movie! It's fantastic.",
    "This film is terrible. I hate it.",
    "What a great experience, I really enjoyed it!",
    "Absolutely awful. Would not recommend."
]
labels = [1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()
y = labels

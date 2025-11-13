from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are great pets.",
    "Dogs love to run more than cats."
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to dense array and retrieve feature names
dense = tfidf_matrix.todense()
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for better visualization
import pandas as pd
df_tfidf = pd.DataFrame(dense, columns=feature_names)

print(df_tfidf)

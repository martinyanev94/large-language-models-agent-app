from transformers import pipeline

# Loading a pre-trained sentiment analysis model
sentiment_pipeline = pipeline('sentiment-analysis')

# Analyzing sentiment of a sentence
result = sentiment_pipeline("I love programming with Python. It's incredibly enjoyable!")
print(result)

def sentiment_analysis(text):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return "No valid words for analysis"
    sentiment = np.mean(vectors, axis=0)
    return sentiment

print(sentiment_analysis('the cat sat'))

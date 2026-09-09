reviews = [
    "I love this movie",
    "This movie is amazing",
    "I hate this movie",
    "This movie is terrible"
]

labels = [
    "positive",
    "positive",
    "negative",
    "negative"
]

vocabulary = ["i", "love", "this", "movie", "amazing", "hate", "terrible"]
text = "i  hate this movie"

words = text.lower().split()
vector = []

for word in vocabulary:
    if word in words:
        vector.append(1)
        
    else:
        vector.append(0)
        
print(vector)            
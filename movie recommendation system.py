import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Movie dataset
movies = pd.DataFrame({
    "title": [
        "Interstellar",
        "The Martian",
        "Gravity",
        "Frozen",
        "Avatar"
    ],
    "genres": [
        "sci-fi space adventure",
        "sci-fi space survival",
        "sci-fi space survival",
        "animation musical family",
        "sci-fi adventure fantasy"
    ]
})

# Convert genres into numerical feature vectors
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(movies["genres"])

# Calculate similarity between all movies
similarity = cosine_similarity(features)

def recommend(movie_name, top_n=3):
    matches = movies.index[
        movies["title"].str.lower() == movie_name.lower()
    ].tolist()

    if not matches:
        print("Movie not found.")
        return

    movie_index = matches[0]

    scores = list(enumerate(similarity[movie_index]))

    # Sort from most similar to least similar
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    for index, score in scores[:top_n]:
        print(movies.iloc[index]["title"], round(score, 2))

recommend("The Martian")
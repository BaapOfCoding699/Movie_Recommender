import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('movies.csv')
print("Heroo, here is your movies data : ")
print(df.head())

cv = CountVectorizer()
vector_matrix = cv.fit_transform(df['genre'])
similarity = cosine_similarity(vector_matrix)
print("\nSimilarity Scores for Iron Man : ")
print(similarity[0])

def get_recommendation(movie_name):
    movie_index = df[df['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True , key = lambda x: x[1])[1:4]
    print(f"\nSince you liked {movie_name} you should watch : ")
    for i in movie_list:
        print(df.iloc[i[0]].title)

get_recommendation("Iron Man")

import pandas as pd
# import warnings 
# warnings.filterwarnings('ignore')
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits,on = 'title')
# print("Dataset Ready")
# print(movies.head(1))

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print("\n Selected features : ")
# print(movies.head())

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
# print("\nClean genres : ")
# print(movies.head())

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","")for i in x])
movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# print("\n Cast and director extracted : ")
# print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id','title','tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print("Tags ready")
# print(movies.head())

cv = CountVectorizer(max_features = 5000 ,stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# print("Vectorizarion Complete : ")
# print(vectors.shape)

similarity = cosine_similarity(vectors)
# print("Ready for recommendation....")

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True , key = lambda x: x[1])[1:6]
    print(f"\nSince you like {movie}, should watch : ")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    
recommend('Avatar')

pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
print("Doneee.....")

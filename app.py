import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(title):
    try:
        url = "https://v3.sg.media-imdb.com/suggestion/x/{}.json".format(title.lower().replace(" ", "%20"))
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        poster_path = data['d'][0]['i']['imageUrl']
        return poster_path
    except:
        return "https://www.prokerala.com/movies/assets/img/no-poster-available.jpg"
                            

movies_dict = pickle.load(open('movie_list.pkl','rb'))
movies = pd.DataFrame(movies_dict)
@st.cache_resource
def get_similarity():
    cv = CountVectorizer(max_features = 5000 ,stop_words = 'english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

similarity = get_similarity()

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True , key = lambda x: x[1])[1:6]
    recommend_movies = []
    recommend_posters = []
    for i in movies_list:
        movie_title = movies.iloc[i[0]].title
        recommend_movies.append(movie_title)
        recommend_posters.append(fetch_poster(movie_title))
    return recommend_movies,recommend_posters

st.title('Movie Recommender System')
selected_movie = st.selectbox('Ladlee, which movie did you watched recently ?',movies['title'].values)

if st.button('Recommend'):
    names , posters = recommend(selected_movie)
    col1 , col2 , col3 , col4 , col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])


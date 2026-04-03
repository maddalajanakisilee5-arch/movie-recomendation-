import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies.csv")

cv = CountVectorizer()
matrix = cv.fit_transform(df['genres'])

similarity = cosine_similarity(matrix)

def recommend(movie):
    if movie not in df['title'].values:
        return ["Movie not found"]

    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))

    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(df.iloc[i[0]].title)

    return recommended_movies

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Get similar movies using Machine Learning")

selected_movie = st.selectbox(
    "Choose a movie",
    df['title'].values
)

if st.button("Recommend"):
    results = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in results:
        st.write("👉", movie)

st.sidebar.title("About")
st.sidebar.write("This app uses ML (Cosine Similarity) to recommend movies.")

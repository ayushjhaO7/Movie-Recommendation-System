import streamlit as st
import pandas as pd
import pickle

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

def reccomend(movie):
    movie_index = movies[movies['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    reccomended_movies = []
    for i in movies_list:
        reccomended_movies.append(movies.iloc[i[0]].original_title)
    
    return reccomended_movies

st.title("Movie Recommendation System")

selected_movie = st.selectbox('Select a movie you like', movies['original_title'].values)

if st.button('Recommend'):
    st.write(f"You selected: {selected_movie}")
    
    # Here you would add the recommendation logic
    final = reccomend(selected_movie)
    for i in final:
        st.write(i)
import streamlit as st
import pandas as pd
import pickle
import requests

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

# api key 8265bd1679663a7ea12ac168da84d2e8
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    response = requests.get(url)
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] 

def reccomend(movie):
    movie_index = movies[movies['original_title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    reccomended_movies = []
    reccomended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        reccomended_movies.append(movies.iloc[i[0]].original_title)
        
        # fetch poster from api
        reccomended_posters.append(fetch_poster(movie_id))
    
    return reccomended_movies , reccomended_posters

st.title("Movie Recommendation System")

selected_movie = st.selectbox('Select a movie you like', movies['original_title'].values)

if st.button('Recommend'):
    st.write(f"You selected: {selected_movie}")
    
    # Here you would add the recommendation logic
    names , posters = reccomend(selected_movie)
    
    col1, col2, col3, col4, col5 = st.columns(5)
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
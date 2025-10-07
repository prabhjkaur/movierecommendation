import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="🎥 Movie Recommendation System", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    h1 {
        text-align: center;
        color: #FF4B4B;
        font-size: 3rem !important;
        font-weight: 700;
        text-shadow: 1px 1px 3px #000;
    }
    .movie-card {
        background-color: #1A1C22;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(255, 75, 75, 0.5);
    }
    .movie-title {
        font-weight: 600;
        color: #FFB86B;
        margin-top: 8px;
        font-size: 1rem;
    }
    .movie-meta {
        font-size: 0.8rem;
        color: #AAAAAA;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies_processed_with_posters.csv")
    df = df.dropna(subset=['soup'])
    return df

df = load_data()

# ---------------------------
# TF-IDF Matrix
# ---------------------------
@st.cache_resource
def create_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = create_similarity(df)

# ---------------------------
# Recommend Function
# ---------------------------
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# ---------------------------
# UI
# ---------------------------
st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("🎞️ Select a movie to get recommendations:", movie_list)

if st.button("🔍 Recommend"):
    with st.spinner("Finding your next binge-worthy picks..."):
        recommendations = recommend(selected_movie)

        if recommendations.empty:
            st.error("No recommendations found.")
        else:
            st.markdown(f"<h3 style='text-align:center;'>Movies similar to <span style='color:#FFB86B;'>{selected_movie}</span></h3>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            cols = st.columns(5)
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % 5]:
                    st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)

                    poster = row['poster_path'] if pd.notna(row['poster_path']) and row['poster_path'] != "" else "https://via.placeholder.com/200x300?text=No+Image"
                    st.image(poster, use_container_width=True)

                    st.markdown(f"<div class='movie-title'>{row['title']}</div>", unsafe_allow_html=True)

                    if 'genres' in row and isinstance(row['genres'], str):
                        genres = ', '.join(eval(row['genres'])[:2]) if row['genres'].startswith('[') else row['genres']
                        st.markdown(f"<div class='movie-meta'>{genres}</div>", unsafe_allow_html=True)

                    if 'director' in row and isinstance(row['director'], str):
                        st.markdown(f"<div class='movie-meta'>👨‍💼 {row['director']}</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

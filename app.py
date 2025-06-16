
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_books_dataset.csv")
    for col in ["book_title", "book_author", "Category", "Summary"]:
        df[col] = df[col].fillna("").astype(str).str.lower()
    df["metadata"] = df["book_title"] + " " + df["book_author"] + " " + df["Category"] + " " + df["Summary"]
    df["clean_title"] = df["book_title"].str.strip()
    return df

df = load_data()

@st.cache_resource
def get_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["metadata"])
    return tfidf_matrix

tfidf_matrix = get_tfidf_matrix(df)
title_to_index = pd.Series(df.index, index=df["clean_title"]).drop_duplicates()

def content_based_recommend(book_title, top_n=5):
    book_title = book_title.lower().strip()
    if book_title not in title_to_index:
        return pd.DataFrame({"book_title": [f"❌ '{book_title}' not found"]})

    idx = title_to_index[book_title]
    query_vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    sim_indices = [i for i in sim_scores.argsort()[::-1] if i != idx][:top_n]

    results = df.iloc[sim_indices][["book_title", "book_author", "Category"]].copy()
    results["Similarity"] = [round(sim_scores[i], 3) for i in sim_indices]
    return results.reset_index(drop=True)

def genre_filter_recommendation(genre, min_rating=0.0, year_range=None, top_n=5):
    filtered = df.copy()
    if genre:
        filtered = filtered[filtered["Category"].str.contains(genre.lower())]
    if year_range:
        filtered = filtered[filtered["year_of_publication"].between(year_range[0], year_range[1], inclusive='both')]
    if min_rating > 0:
        filtered = filtered[filtered["rating"] >= min_rating]

    top = filtered.sort_values(by="rating", ascending=False).drop_duplicates("book_title").head(top_n)
    return top[["book_title", "book_author", "Category", "rating", "year_of_publication"]]

st.title("📖 Personalized Book Recommendation Engine")
st.write("Choose your filters or enter your favorite books to get tailored recommendations.")

col1, col2 = st.columns(2)

with col1:
    fav_books = st.text_input("Enter your favorite books (comma separated)", placeholder="e.g., Twilight, 1984")

with col2:
    genre_input = st.text_input("Preferred Genre", placeholder="e.g., fantasy, romance")

min_rating = st.slider("Minimum Rating", 0.0, 10.0, 3.5, 0.1)
year_range = st.slider("Year of Publication Range", 1900, 2025, (2000, 2022))

top_n = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

if st.button("🔎 Get Recommendations"):
    recommendations = pd.DataFrame()

    if fav_books:
        fav_list = [b.strip().lower() for b in fav_books.split(",") if b.strip()]
        for book in fav_list:
            recs = content_based_recommend(book, top_n=top_n)
            recommendations = pd.concat([recommendations, recs], ignore_index=True)

    if genre_input or recommendations.empty:
        genre_recs = genre_filter_recommendation(genre_input, min_rating, year_range, top_n)
        recommendations = pd.concat([recommendations, genre_recs], ignore_index=True)

    if not recommendations.empty:
        st.subheader("📚 Your Personalized Recommendations")
        st.dataframe(recommendations.drop_duplicates("book_title").head(top_n))
    else:
        st.warning("No recommendations found. Try changing filters or inputs.")

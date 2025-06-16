# 📚 Book Recommendation System

This project is a personalized book recommendation engine built using three popular techniques — Content-Based Filtering, Collaborative Filtering, 
and a Hybrid Model — wrapped in an interactive **Streamlit UI** for seamless exploration.

---

## 🚀 Features

- 🔎 **Content-Based Filtering**: Recommends books similar in genre, author, and summary.
- 👥 **Collaborative Filtering**: Recommends books based on ratings and behavior of similar users.
- ⚖️ **Hybrid Model**: Merges content and collaborative filtering for robust recommendations.
- 📊 **Model Evaluation**: Compared using Precision@5 and Recall@5.
- 🌐 **Streamlit Web App**: Clean, modular interface with interactive filtering.
- ❄️ **Cold Start Support**: Suggests books based on genre when there's no prior user data.

---

🔗 Dataset
Raw Data: https://drive.google.com/file/d/1Bm-IYt7OD-t_5E5p38Lq-RJXt271BlCU/view?usp=sharing

Cleaned Data: https://drive.google.com/file/d/1vzBAtj0xxPElUgOmE4Xz7VqBxmYbF04i/view?usp=sharing



🧠 Models Implemented
1. Content-Based Filtering
Uses TF-IDF vectorization on book metadata (title, author, summary, genres).

Computes cosine similarity to find similar books.

2. Collaborative Filtering
Memory-based approach using user-item rating matrix.

Computes similarity between books based on user ratings.

3. Hybrid Model
Combines both content and collaborative similarities using weighted score.


 Personalization
Accepts user input (favorite books or genres) through a Streamlit UI.

Returns top-N personalized recommendations.

Supports basic cold-start handling for new users.


💻 Tech Stack
Python

Pandas, NumPy, scikit-learn

Streamlit

TQDM

Google Colab / Jupyter Notebook

GitHub



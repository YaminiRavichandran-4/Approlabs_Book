
📊 Summary and Findings Report
Project Title: Approlabs Book Recommendation System
Domain: Recommender Systems / NLP
Tech Stack: Python, scikit-learn, pandas, Streamlit, cosine similarity, collaborative filtering

🧾 Project Overview
This project aims to develop an intelligent Book Recommendation System using three core strategies:

Content-Based Filtering

Collaborative Filtering

Hybrid Recommendation Model

The system helps users discover books they are likely to enjoy based on metadata (like title, author, category, and summary) and user ratings.

🔍 Data Summary
Attribute	Description
book_title	Title of the book
book_author	Author(s) of the book
Category	Genre or category of the book
Summary	Description or plot summary
user_id	User identifier
rating	User's rating for the book (0–10 scale)
Language, Country	Additional metadata

Size:

~1 million records (raw)

~250k after cleaning and deduplication

⚙️ Models Implemented
1. Content-Based Filtering
Approach: TF-IDF on combined metadata (title + author + category + summary)

Similarity: Cosine similarity

Use Case: Ideal for cold-start and when user preferences are known

2. Collaborative Filtering
Approach: Item-item collaborative filtering (memory-based)

Matrix: User-item matrix with rating values

Use Case: Works best when a user has rated many books

3. Hybrid Model
Approach: Weighted average of content and collaborative similarities

Formula: hybrid_score = α * content_score + (1 - α) * collab_score

Result: Combines the strengths of both models

📈 Evaluation Metrics
Model	Precision@5	Recall@5
Content-Based	0.009	0.163
Collaborative	0.540	0.003
Hybrid	8.844	0.289

✅ Hybrid model performed best, with significantly better balance between relevance (precision) and coverage (recall).

🧠 Key Findings
Many books had missing or duplicate metadata; deduplication and cleaning were crucial.

Cosine similarity on summary+genre metadata effectively identifies related books in content-based filtering.

Collaborative filtering struggles with cold-start users but excels for active users.

The hybrid model clearly outperforms individual methods in both precision and recall.

Using 10,000+ popular books keeps memory usage manageable for large-scale computation.




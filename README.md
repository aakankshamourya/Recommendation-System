

# Recommendation-System

## 📌 Project Description

This project is a **Recommendation System** designed to suggest items (such as movies, products, or music) to users based on their preferences or behavior.

It implements **multiple approaches to recommendation**:

1. **Popularity-Based Recommendation**

   * Recommends items that are globally popular (e.g., best-selling products, trending movies).
   * Simple baseline system.

2. **Content-Based Filtering**

   * Uses item features (e.g., genre, description, keywords) to recommend similar items.
   * If a user liked *Movie A*, it suggests other movies with similar features.

3. **Collaborative Filtering (User–Item Interactions)**

   * Uses user ratings or purchase history.
   * Finds patterns (e.g., “Users who liked X also liked Y”).
   * Usually implemented with:

     * **User-based CF** (similar users)
     * **Item-based CF** (similar items)

4. **Model-Based Collaborative Filtering (Matrix Factorization)**

   * Uses techniques like **SVD (Singular Value Decomposition)** or deep learning (e.g., neural collaborative filtering).
   * Learns hidden embeddings of users and items to predict preferences.

---

## 📂 How the Code Works (Typical Flow)

1. **Load Dataset**

   * Reads data (e.g., `ratings.csv`, `movies.csv`, `products.csv`).
   * Example: MovieLens dataset (users, movies, ratings).

   ```python
   import pandas as pd
   ratings = pd.read_csv("ratings.csv")
   movies = pd.read_csv("movies.csv")
   ```

2. **Preprocessing**

   * Handle missing values.
   * Merge datasets (ratings + items).
   * Create user–item interaction matrix.

   ```python
   data = pd.merge(ratings, movies, on="movieId")
   ```

3. **Popularity Model**

   ```python
   top_movies = data.groupby("title")["rating"].mean().sort_values(ascending=False)
   ```

4. **Content-Based Filtering**

   * Use **TF-IDF / CountVectorizer** on movie descriptions.
   * Compute similarity with **cosine similarity**.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity

   tfidf = TfidfVectorizer(stop_words="english")
   tfidf_matrix = tfidf.fit_transform(movies["description"])
   cosine_sim = cosine_similarity(tfidf_matrix)
   ```

5. **Collaborative Filtering (Memory-Based)**

   * Create user–item pivot matrix.
   * Compute similarity (Pearson / cosine).

   ```python
   user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating")
   ```

6. **Model-Based CF (Matrix Factorization)**

   * Use **SVD** from `scipy` or `surprise` library.

   ```python
   from surprise import SVD, Dataset, Reader
   reader = Reader(rating_scale=(0, 5))
   data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
   algo = SVD()
   algo.fit(data.build_full_trainset())
   ```

   * Predict user’s rating for an item:

     ```python
     algo.predict(user_id, movie_id)
     ```

7. **Recommendation Function**

   * Combines one or more methods.
   * Returns top N recommended items.

---

## 🎯 Example Flow for a User

1. User logs in → system checks their history.
2. If **new user** → Popularity-based or content-based recommendations.
3. If **existing user** → Collaborative filtering / matrix factorization.
4. System outputs top N recommendations (e.g., “You may like these 5 movies”).



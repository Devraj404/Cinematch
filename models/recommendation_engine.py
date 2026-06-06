"""
============================================================
Recommendation Engine  – Core ML Module
============================================================
Implements three recommendation strategies:

1. Content-Based Filtering
   – Recommends items similar to what a user has liked,
     based on item features (genre, description, etc.)
   – Uses TF-IDF vectorization + cosine similarity.

2. Collaborative Filtering
   – Recommends items liked by similar users.
   – Builds a user-item rating matrix and uses cosine
     similarity between users.

3. Hybrid System
   – Combines scores from both methods (weighted average)
     to produce the final recommendation list.
============================================================
"""

import numpy  as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise       import cosine_similarity
from sklearn.metrics                import mean_squared_error
from math import sqrt


# ═══════════════════════════════════════════════════════════
# SECTION 1 – Content-Based Filtering
# ═══════════════════════════════════════════════════════════

class ContentBasedFilter:
    """
    Content-Based Filtering using TF-IDF + Cosine Similarity.

    How it works:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Build a "soup" string for each item combining    │
    │    genre, description, and year.                    │
    │ 2. Convert soup strings to TF-IDF vectors.          │
    │ 3. Compute cosine similarity between all items.     │
    │ 4. For a target item, return top-N most similar.    │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.vectorizer      = TfidfVectorizer(stop_words="english")
        self.similarity_matrix = None  # Will be (n_items × n_items) matrix
        self.content_df      = None    # DataFrame of all content

    def fit(self, content_list: list[dict]):
        """
        Train the model with content metadata.

        Args:
            content_list: List of dicts with keys:
                          id, title, genre, description, year
        """
        self.content_df = pd.DataFrame(content_list)

        # Build the "soup": a string combining all features
        # Genre words are repeated to give them extra weight
        self.content_df["soup"] = (
            self.content_df["genre"].str.replace(",", " ") + " " +
            self.content_df["genre"].str.replace(",", " ") + " " +  # repeat genre for weight
            self.content_df["description"].fillna("") + " " +
            self.content_df["year"].astype(str)
        )

        # TF-IDF: converts text to numerical vectors
        # Each word gets a score based on how unique it is
        tfidf_matrix = self.vectorizer.fit_transform(self.content_df["soup"])

        # Cosine similarity: measures angle between vectors
        # Score of 1.0 = identical, 0.0 = completely different
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend(self, content_id: int, top_n: int = 6) -> list[dict]:
        """
        Get top-N items most similar to the given content_id.

        Args:
            content_id: ID of the reference content item
            top_n:      Number of recommendations to return

        Returns:
            List of content dicts sorted by similarity score (desc)
        """
        if self.content_df is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Find the row index for this content_id
        idx_series = self.content_df.index[self.content_df["id"] == content_id]
        if idx_series.empty:
            return []

        idx = idx_series[0]

        # Get similarity scores for all items vs this one
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score (descending), skip the item itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

        # Build result list
        results = []
        for row_idx, score in sim_scores:
            item = self.content_df.iloc[row_idx].to_dict()
            item["similarity_score"] = round(float(score), 4)
            results.append(item)

        return results

    def recommend_for_user(self, liked_ids: list[int], top_n: int = 8) -> list[dict]:
        """
        Recommend content for a user based on their liked items.

        Strategy: Average the similarity scores across all liked items,
        then return the top-N unseen items.

        Args:
            liked_ids: List of content IDs the user has rated highly (≥ 4.0)
            top_n:     Number of recommendations
        """
        if not liked_ids or self.content_df is None:
            return []

        # Start with zero scores for all items
        aggregate_scores = np.zeros(len(self.content_df))

        for cid in liked_ids:
            idx_series = self.content_df.index[self.content_df["id"] == cid]
            if idx_series.empty:
                continue
            idx = idx_series[0]
            aggregate_scores += self.similarity_matrix[idx]

        # Exclude already-liked items
        liked_indices = []
        for cid in liked_ids:
            s = self.content_df.index[self.content_df["id"] == cid]
            if not s.empty:
                liked_indices.append(s[0])

        for i in liked_indices:
            aggregate_scores[i] = -1  # Mark as seen so it won't be recommended

        # Get top-N indices
        top_indices = np.argsort(aggregate_scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            item  = self.content_df.iloc[idx].to_dict()
            item["cb_score"] = round(float(aggregate_scores[idx]), 4)
            results.append(item)

        return results


# ═══════════════════════════════════════════════════════════
# SECTION 2 – Collaborative Filtering
# ═══════════════════════════════════════════════════════════

class CollaborativeFilter:
    """
    User-Based Collaborative Filtering using Cosine Similarity.

    How it works:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Build a user-item rating matrix (users × items). │
    │ 2. Compute cosine similarity between all users.     │
    │ 3. For a target user, find the K most similar users.│
    │ 4. Recommend items those similar users liked that   │
    │    the target user hasn't seen yet.                 │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, k_neighbors: int = 5):
        """
        Args:
            k_neighbors: Number of similar users to consider
        """
        self.k               = k_neighbors
        self.user_item_matrix = None   # DataFrame: rows=users, cols=items
        self.user_similarity  = None   # (n_users × n_users) similarity matrix
        self.user_ids         = []
        self.item_ids         = []

    def fit(self, ratings_list: list[dict]):
        """
        Build the user-item matrix and compute user similarities.

        Args:
            ratings_list: List of dicts with keys: user_id, content_id, rating
        """
        if not ratings_list:
            return

        df = pd.DataFrame(ratings_list)

        # Pivot: rows = users, columns = content items, values = ratings
        # Missing ratings are filled with 0 (unrated)
        self.user_item_matrix = df.pivot_table(
            index="user_id", columns="content_id",
            values="rating", fill_value=0
        )

        self.user_ids = list(self.user_item_matrix.index)
        self.item_ids = list(self.user_item_matrix.columns)

        # Compute cosine similarity between all user vectors
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def recommend(self, user_id: int, all_item_ids: list[int],
                  top_n: int = 8) -> list[dict]:
        """
        Recommend items for a user based on similar users' preferences.

        Args:
            user_id:      ID of the target user
            all_item_ids: Complete list of item IDs in the system
            top_n:        Number of recommendations to return

        Returns:
            List of dicts: [{content_id, cf_score}, ...]
        """
        if self.user_item_matrix is None:
            return []

        # If user has no ratings, can't do CF → return empty
        if user_id not in self.user_ids:
            return []

        user_idx  = self.user_ids.index(user_id)
        user_row  = self.user_item_matrix.iloc[user_idx]

        # Find K most similar users (excluding the user themselves)
        sim_scores   = list(enumerate(self.user_similarity[user_idx]))
        sim_scores   = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_users    = [(i, s) for i, s in sim_scores if i != user_idx][:self.k]

        # Items the current user has already rated
        rated_items = set(user_row[user_row > 0].index)

        # Aggregate weighted ratings from similar users
        item_scores = {}
        total_sim   = {}

        for neighbor_idx, similarity in top_users:
            if similarity <= 0:
                continue
            neighbor_row = self.user_item_matrix.iloc[neighbor_idx]

            for item_id, rating in neighbor_row.items():
                if item_id in rated_items or rating == 0:
                    continue  # Skip already-seen or unrated items

                # Weighted sum: higher-similarity users contribute more
                item_scores[item_id]  = item_scores.get(item_id, 0) + similarity * rating
                total_sim[item_id]    = total_sim.get(item_id, 0) + similarity

        # Normalize scores by total similarity weight
        predicted = {}
        for item_id in item_scores:
            if total_sim[item_id] > 0:
                predicted[item_id] = item_scores[item_id] / total_sim[item_id]

        # Sort by predicted rating (descending) and take top N
        sorted_items = sorted(predicted.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [{"content_id": int(cid), "cf_score": round(score, 4)}
                for cid, score in sorted_items]

    def evaluate(self, test_ratings: list[dict]) -> dict:
        """
        Evaluate the model on test ratings using RMSE.

        RMSE (Root Mean Squared Error): measures how far predictions
        are from actual ratings. Lower = better.

        Args:
            test_ratings: List of dicts with user_id, content_id, rating

        Returns:
            dict with rmse, mae values
        """
        actual    = []
        predicted = []

        for r in test_ratings:
            user_id    = r["user_id"]
            content_id = r["content_id"]

            if user_id not in self.user_ids:
                continue

            user_idx = self.user_ids.index(user_id)
            sim_scores = list(enumerate(self.user_similarity[user_idx]))
            top_users  = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_users  = [(i, s) for i, s in top_users if i != user_idx][:self.k]

            weighted_sum  = 0.0
            total_sim_val = 0.0

            for neighbor_idx, sim in top_users:
                neighbor_row = self.user_item_matrix.iloc[neighbor_idx]
                if content_id in neighbor_row.index and neighbor_row[content_id] > 0:
                    weighted_sum  += sim * neighbor_row[content_id]
                    total_sim_val += sim

            if total_sim_val > 0:
                pred = weighted_sum / total_sim_val
                actual.append(r["rating"])
                predicted.append(pred)

        if not actual:
            return {"rmse": None, "mae": None, "samples": 0}

        rmse = sqrt(mean_squared_error(actual, predicted))
        mae  = np.mean(np.abs(np.array(actual) - np.array(predicted)))

        return {"rmse": round(rmse, 4), "mae": round(mae, 4),
                "samples": len(actual)}


# ═══════════════════════════════════════════════════════════
# SECTION 3 – Hybrid Recommendation Engine
# ═══════════════════════════════════════════════════════════

class HybridRecommender:
    """
    Hybrid Recommendation System.

    Combines Content-Based and Collaborative Filtering scores
    using a weighted average:

        final_score = α × CF_score + (1-α) × CB_score

    where α (alpha) controls how much CF contributes.
    Default α = 0.6 (slightly favor CF as it's more personalized).

    Cold Start Problem:
    – New users (no ratings): α = 0.0 → pure content-based
    – New items (no ratings): recommend based on user genre prefs
    """

    def __init__(self, alpha: float = 0.6):
        """
        Args:
            alpha: Weight given to CF (0.0 = pure CB, 1.0 = pure CF)
        """
        self.alpha = alpha
        self.cb    = ContentBasedFilter()
        self.cf    = CollaborativeFilter(k_neighbors=5)
        self._fitted = False

    def fit(self, content_list: list[dict], ratings_list: list[dict]):
        """
        Train both models.

        Args:
            content_list: All content metadata dicts
            ratings_list: All rating records dicts
        """
        self.cb.fit(content_list)
        self.cf.fit(ratings_list)
        self._fitted = True

    def recommend(self, user_id: int, user_ratings: list[dict],
                  all_content: list[dict], top_n: int = 8) -> list[dict]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id:      Target user's ID
            user_ratings: User's historical ratings [{content_id, rating}]
            all_content:  Full content metadata list
            top_n:        How many recommendations to return

        Returns:
            Ranked list of content dicts with scores
        """
        all_item_ids = [c["id"] for c in all_content]
        rated_ids    = {r["content_id"] for r in user_ratings}

        # ── Cold Start: new user with no ratings ─────────────
        if len(user_ratings) < 3:
            # Fall back to trending / highest-rated content
            unrated = [c for c in all_content if c["id"] not in rated_ids]
            unrated_sorted = sorted(unrated,
                                    key=lambda x: x.get("avg_rating", 0),
                                    reverse=True)
            for item in unrated_sorted[:top_n]:
                item["recommendation_type"] = "trending (cold start)"
            return unrated_sorted[:top_n]

        # ── Content-Based scores ─────────────────────────────
        # Use items rated ≥ 4.0 as "liked" seeds
        liked_ids = [r["content_id"] for r in user_ratings if r["rating"] >= 4.0]
        cb_recs   = self.cb.recommend_for_user(liked_ids, top_n=top_n * 2)
        cb_scores = {item["id"]: item.get("cb_score", 0) for item in cb_recs}

        # ── Collaborative Filtering scores ───────────────────
        cf_recs   = self.cf.recommend(user_id, all_item_ids, top_n=top_n * 2)
        cf_scores = {item["content_id"]: item.get("cf_score", 0)
                     for item in cf_recs}

        # ── Normalize scores to [0, 1] range ─────────────────
        def normalize(scores: dict) -> dict:
            if not scores:
                return {}
            max_v = max(scores.values()) or 1
            return {k: v / max_v for k, v in scores.items()}

        cb_norm = normalize(cb_scores)
        cf_norm = normalize(cf_scores)

        # ── Combine: weighted average ─────────────────────────
        candidate_ids = set(cb_norm.keys()) | set(cf_norm.keys())
        combined = {}
        for cid in candidate_ids:
            if cid in rated_ids:
                continue  # Don't recommend already-seen content
            cb_val = cb_norm.get(cid, 0)
            cf_val = cf_norm.get(cid, 0)
            combined[cid] = self.alpha * cf_val + (1 - self.alpha) * cb_val

        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Attach full content metadata to results
        content_map = {c["id"]: c for c in all_content}
        results = []
        for cid, score in ranked:
            if cid in content_map:
                item = content_map[cid].copy()
                item["hybrid_score"]          = round(score, 4)
                item["cb_score"]              = round(cb_norm.get(cid, 0), 4)
                item["cf_score"]              = round(cf_norm.get(cid, 0), 4)
                item["recommendation_type"]   = "hybrid"
                results.append(item)

        return results


# ═══════════════════════════════════════════════════════════
# SECTION 4 – Evaluation Metrics
# ═══════════════════════════════════════════════════════════

class RecommenderEvaluator:
    """
    Evaluation metrics for recommendation systems.

    Metrics explained:
    ──────────────────
    Precision@K: Of the top-K recommendations, what fraction
                 did the user actually like?
                 → High precision = few irrelevant recommendations

    Recall@K:    Of all items the user actually likes, what
                 fraction appeared in the top-K recs?
                 → High recall = not missing relevant items

    F1@K:        Harmonic mean of Precision and Recall.
                 Balances both metrics into a single score.

    RMSE:        Root Mean Squared Error for rating prediction.
                 Measures accuracy of predicted vs actual ratings.
    """

    @staticmethod
    def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
        """
        Precision@K: Fraction of top-K recs that are relevant.

        Args:
            recommended_ids: Ordered list of recommended content IDs
            relevant_ids:    Set of content IDs the user actually likes
            k:               Cut-off rank

        Returns:
            Precision score in [0.0, 1.0]
        """
        top_k = recommended_ids[:k]
        hits  = sum(1 for rid in top_k if rid in relevant_ids)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
        """
        Recall@K: Fraction of relevant items that appear in top-K.

        Args: same as precision_at_k
        Returns: Recall score in [0.0, 1.0]
        """
        top_k = recommended_ids[:k]
        hits  = sum(1 for rid in top_k if rid in relevant_ids)
        return hits / len(relevant_ids) if relevant_ids else 0.0

    @staticmethod
    def f1_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
        """
        F1@K: Harmonic mean of Precision@K and Recall@K.
        """
        p = RecommenderEvaluator.precision_at_k(recommended_ids, relevant_ids, k)
        r = RecommenderEvaluator.recall_at_k(recommended_ids, relevant_ids, k)
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def rmse(actual: list, predicted: list) -> float:
        """
        RMSE: Root Mean Squared Error.
        Commonly used to evaluate rating prediction accuracy.
        """
        if not actual:
            return float("nan")
        return sqrt(mean_squared_error(actual, predicted))

    @staticmethod
    def evaluate_all(recommended_ids: list, relevant_ids: set,
                     actual_ratings: list = None,
                     predicted_ratings: list = None) -> dict:
        """
        Run all metrics and return a summary report.
        """
        k = min(10, len(recommended_ids))
        report = {
            f"precision@{k}": RecommenderEvaluator.precision_at_k(
                recommended_ids, relevant_ids, k),
            f"recall@{k}": RecommenderEvaluator.recall_at_k(
                recommended_ids, relevant_ids, k),
            f"f1@{k}": RecommenderEvaluator.f1_at_k(
                recommended_ids, relevant_ids, k),
        }
        if actual_ratings and predicted_ratings:
            report["rmse"] = RecommenderEvaluator.rmse(
                actual_ratings, predicted_ratings)
        return report

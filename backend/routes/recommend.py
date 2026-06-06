"""
============================================================
Recommendation Routes  (/api/recommend/...)
============================================================
Exposes the recommendation engine via REST API endpoints.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models import Content, Rating, User
from models.recommendation_engine import HybridRecommender, RecommenderEvaluator

recommend_bp = Blueprint("recommend", __name__)

# ── Module-level recommender instance ────────────────────────
# In production, persist and retrain this on a schedule (e.g. nightly).
_recommender = HybridRecommender(alpha=0.6)
_model_trained = False


def _get_training_data():
    """Fetch all content and ratings from the database."""
    content_list = [c.to_dict() for c in Content.query.all()]
    ratings_list = [{"user_id":    r.user_id,
                     "content_id": r.content_id,
                     "rating":     r.rating}
                    for r in Rating.query.all()]
    return content_list, ratings_list


def _ensure_model_trained():
    """Train the recommender if it hasn't been trained yet."""
    global _model_trained
    if not _model_trained:
        content_list, ratings_list = _get_training_data()
        if content_list:
            _recommender.fit(content_list, ratings_list)
            _model_trained = True


# ── GET /api/recommend/for-me ────────────────────────────────
@recommend_bp.route("/for-me", methods=["GET"])
@jwt_required()
def get_my_recommendations():
    """
    Return personalized recommendations for the logged-in user.

    The hybrid model combines:
    – Content-based: items similar to what you've liked
    – Collaborative: items liked by users similar to you

    Query params:
        top_n: int – number of recommendations (default 8)
        algo:  str – 'hybrid' | 'content' | 'collaborative'
    """
    user_id = int(get_jwt_identity())
    top_n   = request.args.get("top_n", 8, type=int)
    algo    = request.args.get("algo", "hybrid")

    _ensure_model_trained()

    # Fetch user's rating history
    user_ratings_db = Rating.query.filter_by(user_id=user_id).all()
    user_ratings    = [{"content_id": r.content_id, "rating": r.rating}
                       for r in user_ratings_db]

    all_content = [c.to_dict() for c in Content.query.all()]
    rated_ids   = {r["content_id"] for r in user_ratings}

    # ── Route to the chosen algorithm ────────────────────────
    if algo == "content":
        liked_ids = [r["content_id"] for r in user_ratings if r["rating"] >= 4.0]
        recs      = _recommender.cb.recommend_for_user(liked_ids, top_n=top_n)
        for r in recs:
            r["recommendation_type"] = "content-based"

    elif algo == "collaborative":
        cf_recs     = _recommender.cf.recommend(user_id, [c["id"] for c in all_content], top_n)
        content_map = {c["id"]: c for c in all_content}
        recs = []
        for cf in cf_recs:
            cid = cf["content_id"]
            if cid in content_map:
                item = content_map[cid].copy()
                item["cf_score"] = cf["cf_score"]
                item["recommendation_type"] = "collaborative"
                recs.append(item)

    else:  # hybrid (default)
        recs = _recommender.recommend(
            user_id=user_id,
            user_ratings=user_ratings,
            all_content=all_content,
            top_n=top_n,
        )

    return jsonify({
        "algorithm":       algo,
        "recommendations": recs,
        "user_ratings":    len(user_ratings),
    }), 200


# ── GET /api/recommend/similar/<id> ──────────────────────────
@recommend_bp.route("/similar/<int:content_id>", methods=["GET"])
def get_similar(content_id):
    """
    Return items similar to the given content (content-based).
    Useful for "More like this" sections on content detail pages.
    """
    top_n = request.args.get("top_n", 6, type=int)
    _ensure_model_trained()

    similar = _recommender.cb.recommend(content_id=content_id, top_n=top_n)
    return jsonify(similar), 200


# ── POST /api/recommend/retrain ──────────────────────────────
@recommend_bp.route("/retrain", methods=["POST"])
@jwt_required()
def retrain_model():
    """
    Retrain the recommendation model with latest data.
    Should be called after significant new ratings are added.
    In production, automate this with a scheduled job (cron/Celery).
    """
    global _model_trained
    content_list, ratings_list = _get_training_data()
    _recommender.fit(content_list, ratings_list)
    _model_trained = True

    return jsonify({
        "message":       "Model retrained successfully",
        "content_count": len(content_list),
        "ratings_count": len(ratings_list),
    }), 200


# ── GET /api/recommend/evaluate ──────────────────────────────
@recommend_bp.route("/evaluate", methods=["GET"])
def evaluate_model():
    """
    Run evaluation metrics on the recommendation model.

    Uses a simple leave-one-out strategy:
    – For the demo user, hold out their last rating as the test item
    – Generate recommendations and check if the held-out item appears

    Returns Precision, Recall, F1 scores.
    """
    _ensure_model_trained()

    # Find the user with the most ratings for evaluation
    from sqlalchemy import func
    top_user = (db.session.query(Rating.user_id, func.count(Rating.id).label("cnt"))
                .group_by(Rating.user_id)
                .order_by(func.count(Rating.id).desc())
                .first())

    if not top_user:
        return jsonify({"error": "Not enough data for evaluation"}), 400

    user_id = top_user[0]
    all_ratings  = Rating.query.filter_by(user_id=user_id).all()
    all_content  = [c.to_dict() for c in Content.query.all()]

    # Split: use all-but-last ratings for training context, hold out last one
    train_ratings = [{"content_id": r.content_id, "rating": r.rating}
                     for r in all_ratings[:-1]]
    test_item_id  = all_ratings[-1].content_id

    # Generate recommendations using training data only
    recs = _recommender.recommend(
        user_id=user_id,
        user_ratings=train_ratings,
        all_content=all_content,
        top_n=10,
    )

    recommended_ids = [r["id"] for r in recs if "id" in r]
    relevant_ids    = {test_item_id}  # The held-out item is "relevant"

    metrics = RecommenderEvaluator.evaluate_all(recommended_ids, relevant_ids)
    metrics["note"] = "Leave-one-out evaluation on demo user"

    return jsonify(metrics), 200

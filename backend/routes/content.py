"""
============================================================
Content Routes  (/api/content/...)
============================================================
Handles browsing, searching, and rating content (movies).
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models import Content, Rating, User

content_bp = Blueprint("content", __name__)


# ── GET /api/content/ ────────────────────────────────────────
@content_bp.route("/", methods=["GET"])
def get_all_content():
    """
    Return paginated list of all content.

    Query params:
        page    : int  – page number (default 1)
        per_page: int  – items per page (default 12)
        genre   : str  – filter by genre (optional)
        search  : str  – search title (optional)
    """
    page     = request.args.get("page",     1,  type=int)
    per_page = request.args.get("per_page", 12, type=int)
    genre    = request.args.get("genre",    "")
    search   = request.args.get("search",   "")

    query = Content.query

    # Apply filters
    if genre:
        query = query.filter(Content.genre.contains(genre))
    if search:
        query = query.filter(Content.title.ilike(f"%{search}%"))

    # Paginate results
    paginated = query.order_by(Content.avg_rating.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "items":       [c.to_dict() for c in paginated.items],
        "total":       paginated.total,
        "pages":       paginated.pages,
        "current_page": page,
    }), 200


# ── GET /api/content/trending ────────────────────────────────
@content_bp.route("/trending", methods=["GET"])
def get_trending():
    """
    Return top 10 highest-rated content items.
    'Trending' = high avg_rating AND high rating_count.
    """
    trending = (Content.query
                .filter(Content.rating_count >= 10)
                .order_by(Content.avg_rating.desc())
                .limit(10).all())

    return jsonify([c.to_dict() for c in trending]), 200


# ── GET /api/content/<id> ────────────────────────────────────
@content_bp.route("/<int:content_id>", methods=["GET"])
def get_content(content_id):
    """Return details for a single content item."""
    item = Content.query.get_or_404(content_id)
    return jsonify(item.to_dict()), 200


# ── POST /api/content/<id>/rate ──────────────────────────────
@content_bp.route("/<int:content_id>/rate", methods=["POST"])
@jwt_required()
def rate_content(content_id):
    """
    Submit or update a rating for a content item.

    Request body (JSON):
        rating: float  – value between 1.0 and 5.0

    This rating data is used to train/update the collaborative
    filtering model.
    """
    user_id = int(get_jwt_identity())
    data    = request.get_json()
    rating_value = float(data.get("rating", 0))

    # Validate rating range
    if not (1.0 <= rating_value <= 5.0):
        return jsonify({"error": "Rating must be between 1 and 5"}), 400

    # Check if user already rated this content
    existing = Rating.query.filter_by(
        user_id=user_id, content_id=content_id
    ).first()

    if existing:
        existing.rating = rating_value  # Update existing rating
    else:
        # Create new rating record
        new_rating = Rating(user_id=user_id, content_id=content_id,
                            rating=rating_value)
        db.session.add(new_rating)

    # ── Recalculate average rating for this content ───────────
    content = Content.query.get_or_404(content_id)
    all_ratings = Rating.query.filter_by(content_id=content_id).all()
    content.avg_rating   = sum(r.rating for r in all_ratings) / len(all_ratings)
    content.rating_count = len(all_ratings)

    db.session.commit()
    return jsonify({"message": "Rating submitted!", "new_avg": content.avg_rating}), 200


# ── GET /api/content/genres ──────────────────────────────────
@content_bp.route("/genres", methods=["GET"])
def get_genres():
    """Return a sorted list of all unique genres in the database."""
    all_content = Content.query.all()
    genres = set()
    for item in all_content:
        for g in item.genre.split(","):
            genres.add(g.strip())
    return jsonify(sorted(list(genres))), 200


# ── GET /api/content/user-ratings ───────────────────────────
@content_bp.route("/user-ratings", methods=["GET"])
@jwt_required()
def get_user_ratings():
    """Return all ratings submitted by the current user."""
    user_id = int(get_jwt_identity())
    ratings = Rating.query.filter_by(user_id=user_id).all()
    result  = []
    for r in ratings:
        d = r.to_dict()
        d["content_title"] = r.content.title
        result.append(d)
    return jsonify(result), 200

"""
============================================================
Admin Routes  (/api/admin/...)
============================================================
Protected endpoints for managing content and viewing stats.
Only accessible by users with is_admin = True.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models import Content, Rating, User

admin_bp = Blueprint("admin", __name__)


def _require_admin():
    """Helper: raise 403 if the current user is not an admin."""
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return None, jsonify({"error": "Admin access required"}), 403
    return user, None, None


# ── POST /api/admin/content ──────────────────────────────────
@admin_bp.route("/content", methods=["POST"])
@jwt_required()
def add_content():
    """
    Add a new movie/content item to the database.

    Request body (JSON):
        title, genre, year, description, poster (URL)
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.get_json()
    required = ["title", "genre", "year"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    content = Content(
        title       = data["title"],
        genre       = data["genre"],
        year        = int(data["year"]),
        description = data.get("description", ""),
        poster      = data.get("poster", ""),
    )
    db.session.add(content)
    db.session.commit()

    return jsonify({"message": "Content added!", "content": content.to_dict()}), 201


# ── DELETE /api/admin/content/<id> ───────────────────────────
@admin_bp.route("/content/<int:content_id>", methods=["DELETE"])
@jwt_required()
def delete_content(content_id):
    """Delete a content item and all its ratings."""
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    content = Content.query.get_or_404(content_id)
    Rating.query.filter_by(content_id=content_id).delete()
    db.session.delete(content)
    db.session.commit()
    return jsonify({"message": f"'{content.title}' deleted"}), 200


# ── GET /api/admin/stats ─────────────────────────────────────
@admin_bp.route("/stats", methods=["GET"])
@jwt_required()
def get_stats():
    """
    Return system-wide statistics for the admin dashboard.
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    total_users   = User.query.count()
    total_content = Content.query.count()
    total_ratings = Rating.query.count()
    avg_rating    = db.session.query(db.func.avg(Rating.rating)).scalar() or 0

    # Genre distribution
    all_content = Content.query.all()
    genre_counts = {}
    for c in all_content:
        for g in c.genre.split(","):
            g = g.strip()
            genre_counts[g] = genre_counts.get(g, 0) + 1

    return jsonify({
        "total_users":   total_users,
        "total_content": total_content,
        "total_ratings": total_ratings,
        "avg_rating":    round(float(avg_rating), 2),
        "genre_distribution": genre_counts,
    }), 200


# ── GET /api/admin/users ─────────────────────────────────────
@admin_bp.route("/users", methods=["GET"])
@jwt_required()
def get_users():
    """Return list of all users (admin only)."""
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user or not user.is_admin:
        return jsonify({"error": "Admin access required"}), 403

    users = User.query.all()
    result = []
    for u in users:
        d = u.to_dict()
        d["rating_count"] = Rating.query.filter_by(user_id=u.id).count()
        result.append(d)
    return jsonify(result), 200

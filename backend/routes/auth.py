"""
============================================================
Authentication Routes  (/api/auth/...)
============================================================
Handles user registration, login, and profile management.
Uses JWT tokens for stateless authentication.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity
)
from backend.extensions import db, bcrypt
from backend.models import User

auth_bp = Blueprint("auth", __name__)


# ── POST /api/auth/register ──────────────────────────────────
@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user account.

    Request body (JSON):
        username        : str  – unique display name
        email           : str  – unique email address
        password        : str  – plain-text password (hashed before storage)
        preferred_genres: str  – comma-separated e.g. "Action,Drama"

    Returns:
        201 – {"message": "...", "user": {...}}
        400 – {"error": "..."} on validation failure
    """
    data = request.get_json()

    # ── Validate required fields ─────────────────────────────
    required = ["username", "email", "password"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    # ── Check for existing username/email ────────────────────
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username already taken"}), 400
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email already registered"}), 400

    # ── Hash password and create user ────────────────────────
    hashed_pw = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
    user = User(
        username         = data["username"],
        email            = data["email"],
        password_hash    = hashed_pw,
        preferred_genres = data.get("preferred_genres", ""),
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Account created successfully!", "user": user.to_dict()}), 201


# ── POST /api/auth/login ─────────────────────────────────────
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate a user and return a JWT access token.

    Request body (JSON):
        email   : str
        password: str

    Returns:
        200 – {"access_token": "...", "user": {...}}
        401 – {"error": "Invalid credentials"}
    """
    data = request.get_json()
    user = User.query.filter_by(email=data.get("email")).first()

    # Verify user exists and password matches
    if not user or not bcrypt.check_password_hash(user.password_hash, data.get("password", "")):
        return jsonify({"error": "Invalid email or password"}), 401

    # Create JWT token valid for 24 hours
    token = create_access_token(identity=str(user.id))
    return jsonify({"access_token": token, "user": user.to_dict()}), 200


# ── GET /api/auth/profile ────────────────────────────────────
@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    """
    Return the current user's profile.
    Requires a valid JWT in the Authorization header.
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get_or_404(user_id)
    return jsonify(user.to_dict()), 200


# ── PUT /api/auth/profile ────────────────────────────────────
@auth_bp.route("/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    """
    Update the current user's preferred genres.
    This data feeds the content-based filtering model.
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get_or_404(user_id)
    data    = request.get_json()

    if "preferred_genres" in data:
        user.preferred_genres = data["preferred_genres"]

    db.session.commit()
    return jsonify({"message": "Profile updated", "user": user.to_dict()}), 200

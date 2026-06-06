"""
============================================================
Database Models
============================================================
Defines all SQLAlchemy ORM models (tables) for the system.
Each class maps to a database table.
"""

from datetime import datetime
from backend.extensions import db


class User(db.Model):
    """
    Stores registered user information.
    Each user can rate content and receive recommendations.
    """
    __tablename__ = "users"

    id               = db.Column(db.Integer, primary_key=True)
    username         = db.Column(db.String(80),  unique=True, nullable=False)
    email            = db.Column(db.String(120), unique=True, nullable=False)
    password_hash    = db.Column(db.String(200), nullable=False)
    preferred_genres = db.Column(db.String(200), default="")  # comma-separated
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin         = db.Column(db.Boolean, default=False)

    # Relationships
    ratings = db.relationship("Rating", backref="user", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "preferred_genres": self.preferred_genres,
            "created_at": self.created_at.isoformat(),
            "is_admin": self.is_admin,
        }


class Content(db.Model):
    """
    Stores movie/content metadata.
    This is what gets recommended to users.
    """
    __tablename__ = "content"

    id           = db.Column(db.Integer, primary_key=True)
    title        = db.Column(db.String(200), nullable=False)
    genre        = db.Column(db.String(200), nullable=False)   # e.g. "Action,Drama"
    year         = db.Column(db.Integer)
    description  = db.Column(db.Text, default="")
    poster       = db.Column(db.String(500), default="")
    avg_rating   = db.Column(db.Float, default=0.0)
    rating_count = db.Column(db.Integer, default=0)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    ratings = db.relationship("Rating", backref="content", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "genre": self.genre,
            "year": self.year,
            "description": self.description,
            "poster": self.poster,
            "avg_rating": round(self.avg_rating, 2),
            "rating_count": self.rating_count,
        }


class Rating(db.Model):
    """
    Records a user's rating of a piece of content.
    This is the core interaction data used by the ML models.
    Rating scale: 1.0 (worst) to 5.0 (best).
    """
    __tablename__ = "ratings"

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey("content.id"), nullable=False)
    rating     = db.Column(db.Float, nullable=False)   # 1.0 – 5.0
    rated_at   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "content_id": self.content_id,
            "rating": self.rating,
            "rated_at": self.rated_at.isoformat(),
        }

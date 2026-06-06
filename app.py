"""
import sys

import os

app.config["SECRET_KEY"]              = os.environ.get("SECRET_KEY", "fallback-secret")
app.config["JWT_SECRET_KEY"]          = os.environ.get("JWT_SECRET_KEY", "fallback-jwt")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///recommendation.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_cors import CORS


def create_app():
    app = Flask(__name__,
                template_folder="frontend/templates",
                static_folder="frontend/static")

    app.config["SECRET_KEY"]                     = "your-secret-key-change-in-production"
    app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///recommendation.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JWT_SECRET_KEY"]                 = "jwt-secret-change-in-production"

    # ── Import extensions AFTER app config ──────────────────
    from backend.extensions import db, bcrypt, jwt
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    CORS(app)

    # ── Register Blueprints ──────────────────────────────────
    from backend.routes.auth      import auth_bp
    from backend.routes.content   import content_bp
    from backend.routes.recommend import recommend_bp
    from backend.routes.admin     import admin_bp

    app.register_blueprint(auth_bp,      url_prefix="/api/auth")
    app.register_blueprint(content_bp,   url_prefix="/api/content")
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(admin_bp,     url_prefix="/api/admin")

    # ── Frontend routes ──────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/admin")
    def admin():
        return render_template("admin.html")

    # ── Create DB tables and seed data ───────────────────────
    with app.app_context():
        db.create_all()
        _seed_sample_data(db, bcrypt)

    return app


def _seed_sample_data(db, bcrypt):
    from backend.models import User, Content, Rating
    import random

    if Content.query.count() > 0:
        return

    sample_movies = [
        {"title": "The Dark Knight",         "genre": "Action,Crime,Drama",      "year": 2008, "description": "Batman faces the Joker in Gotham City.",         "poster": "https://picsum.photos/seed/dk/300/450"},
        {"title": "Inception",               "genre": "Action,Adventure,Sci-Fi", "year": 2010, "description": "A thief enters dreams to plant an idea.",         "poster": "https://picsum.photos/seed/inc/300/450"},
        {"title": "Interstellar",            "genre": "Adventure,Drama,Sci-Fi",  "year": 2014, "description": "Explorers travel through a wormhole.",            "poster": "https://picsum.photos/seed/int/300/450"},
        {"title": "The Matrix",              "genre": "Action,Sci-Fi",           "year": 1999, "description": "A hacker discovers reality is a simulation.",     "poster": "https://picsum.photos/seed/mat/300/450"},
        {"title": "Pulp Fiction",            "genre": "Crime,Drama",             "year": 1994, "description": "Interlocking stories of crime in LA.",            "poster": "https://picsum.photos/seed/pf/300/450"},
        {"title": "The Shawshank Redemption","genre": "Drama",                   "year": 1994, "description": "Two imprisoned men bond over years.",             "poster": "https://picsum.photos/seed/sr/300/450"},
        {"title": "Forrest Gump",            "genre": "Drama,Romance",           "year": 1994, "description": "Slow-witted Forrest Gump witnesses history.",     "poster": "https://picsum.photos/seed/fg/300/450"},
        {"title": "The Godfather",           "genre": "Crime,Drama",             "year": 1972, "description": "The patriarch of a crime dynasty.",               "poster": "https://picsum.photos/seed/gf/300/450"},
        {"title": "Avengers: Endgame",       "genre": "Action,Adventure,Sci-Fi", "year": 2019, "description": "The Avengers assemble to reverse Thanos.",        "poster": "https://picsum.photos/seed/av/300/450"},
        {"title": "Parasite",                "genre": "Comedy,Drama,Thriller",   "year": 2019, "description": "A poor family schemes into a wealthy home.",       "poster": "https://picsum.photos/seed/par/300/450"},
        {"title": "Get Out",                 "genre": "Horror,Mystery,Thriller", "year": 2017, "description": "A Black man meets his girlfriend's family.",      "poster": "https://picsum.photos/seed/go/300/450"},
        {"title": "Mad Max: Fury Road",      "genre": "Action,Adventure,Sci-Fi", "year": 2015, "description": "Post-apocalyptic chase across the wasteland.",    "poster": "https://picsum.photos/seed/mm/300/450"},
        {"title": "La La Land",              "genre": "Comedy,Drama,Music",      "year": 2016, "description": "A jazz musician and actress fall in love.",        "poster": "https://picsum.photos/seed/ll/300/450"},
        {"title": "Whiplash",                "genre": "Drama,Music",             "year": 2014, "description": "A drummer pushes himself to perfection.",         "poster": "https://picsum.photos/seed/wh/300/450"},
        {"title": "The Social Network",      "genre": "Biography,Drama",         "year": 2010, "description": "The founding of Facebook.",                       "poster": "https://picsum.photos/seed/tsn/300/450"},
        {"title": "Arrival",                 "genre": "Drama,Mystery,Sci-Fi",    "year": 2016, "description": "A linguist communicates with aliens.",             "poster": "https://picsum.photos/seed/arr/300/450"},
        {"title": "Blade Runner 2049",       "genre": "Action,Drama,Mystery",    "year": 2017, "description": "A blade runner uncovers a secret.",               "poster": "https://picsum.photos/seed/br/300/450"},
        {"title": "1917",                    "genre": "Drama,War",               "year": 2019, "description": "Two soldiers race to deliver a message.",         "poster": "https://picsum.photos/seed/1917/300/450"},
        {"title": "Spirited Away",           "genre": "Animation,Adventure",     "year": 2001, "description": "A girl enters a spirit world.",                   "poster": "https://picsum.photos/seed/sa/300/450"},
        {"title": "Joker",                   "genre": "Crime,Drama,Thriller",    "year": 2019, "description": "The origin story of the Joker.",                 "poster": "https://picsum.photos/seed/jok/300/450"},
    ]

    for m in sample_movies:
        content = Content(**m,
                          avg_rating=round(random.uniform(3.5, 5.0), 1),
                          rating_count=random.randint(50, 500))
        db.session.add(content)

    demo = User(
        username="demo",
        email="demo@example.com",
        password_hash=bcrypt.generate_password_hash("password123").decode("utf-8"),
        preferred_genres="Action,Drama,Sci-Fi"
    )
    db.session.add(demo)
    db.session.commit()

    movies = Content.query.all()
    for movie in movies[:10]:
        rating = Rating(user_id=demo.id, content_id=movie.id,
                        rating=random.uniform(1, 5))
        db.session.add(rating)
    db.session.commit()
    print("✅ Sample data seeded successfully.")


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)"""


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template
from flask_cors import CORS


def create_app():
    app = Flask(__name__,
                template_folder="frontend/templates",
                static_folder="frontend/static")

    # ── Configuration ────────────────────────────────────────
    app.config["SECRET_KEY"]                     = os.environ.get("SECRET_KEY", "fallback-secret")
    app.config["JWT_SECRET_KEY"]                 = os.environ.get("JWT_SECRET_KEY", "fallback-jwt")
    app.config["SQLALCHEMY_DATABASE_URI"]        = os.environ.get("DATABASE_URL", "sqlite:///recommendation.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ── Extensions ───────────────────────────────────────────
    from backend.extensions import db, bcrypt, jwt
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    CORS(app)

    # ── Blueprints ───────────────────────────────────────────
    from backend.routes.auth      import auth_bp
    from backend.routes.content   import content_bp
    from backend.routes.recommend import recommend_bp
    from backend.routes.admin     import admin_bp

    app.register_blueprint(auth_bp,      url_prefix="/api/auth")
    app.register_blueprint(content_bp,   url_prefix="/api/content")
    app.register_blueprint(recommend_bp, url_prefix="/api/recommend")
    app.register_blueprint(admin_bp,     url_prefix="/api/admin")

    # ── Frontend Routes ──────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.route("/admin")
    def admin():
        return render_template("admin.html")

    # ── Database Setup ───────────────────────────────────────
    with app.app_context():
        db.create_all()
        _seed_sample_data(db, bcrypt)

    return app


def _seed_sample_data(db, bcrypt):
    from backend.models import User, Content, Rating
    import random

    if Content.query.count() > 0:
        return

    sample_movies = [
        {"title": "The Dark Knight",         "genre": "Action,Crime,Drama",      "year": 2008, "description": "Batman faces the Joker in Gotham City.",         "poster": "https://picsum.photos/seed/dk/300/450"},
        {"title": "Inception",               "genre": "Action,Adventure,Sci-Fi", "year": 2010, "description": "A thief enters dreams to plant an idea.",         "poster": "https://picsum.photos/seed/inc/300/450"},
        {"title": "Interstellar",            "genre": "Adventure,Drama,Sci-Fi",  "year": 2014, "description": "Explorers travel through a wormhole.",            "poster": "https://picsum.photos/seed/int/300/450"},
        {"title": "The Matrix",              "genre": "Action,Sci-Fi",           "year": 1999, "description": "A hacker discovers reality is a simulation.",     "poster": "https://picsum.photos/seed/mat/300/450"},
        {"title": "Pulp Fiction",            "genre": "Crime,Drama",             "year": 1994, "description": "Interlocking stories of crime in LA.",            "poster": "https://picsum.photos/seed/pf/300/450"},
        {"title": "The Shawshank Redemption","genre": "Drama",                   "year": 1994, "description": "Two imprisoned men bond over years.",             "poster": "https://picsum.photos/seed/sr/300/450"},
        {"title": "Forrest Gump",            "genre": "Drama,Romance",           "year": 1994, "description": "Slow-witted Forrest Gump witnesses history.",     "poster": "https://picsum.photos/seed/fg/300/450"},
        {"title": "The Godfather",           "genre": "Crime,Drama",             "year": 1972, "description": "The patriarch of a crime dynasty.",               "poster": "https://picsum.photos/seed/gf/300/450"},
        {"title": "Avengers: Endgame",       "genre": "Action,Adventure,Sci-Fi", "year": 2019, "description": "The Avengers assemble to reverse Thanos.",        "poster": "https://picsum.photos/seed/av/300/450"},
        {"title": "Parasite",                "genre": "Comedy,Drama,Thriller",   "year": 2019, "description": "A poor family schemes into a wealthy home.",       "poster": "https://picsum.photos/seed/par/300/450"},
        {"title": "Get Out",                 "genre": "Horror,Mystery,Thriller", "year": 2017, "description": "A Black man meets his girlfriend's family.",      "poster": "https://picsum.photos/seed/go/300/450"},
        {"title": "Mad Max: Fury Road",      "genre": "Action,Adventure,Sci-Fi", "year": 2015, "description": "Post-apocalyptic chase across the wasteland.",    "poster": "https://picsum.photos/seed/mm/300/450"},
        {"title": "La La Land",              "genre": "Comedy,Drama,Music",      "year": 2016, "description": "A jazz musician and actress fall in love.",        "poster": "https://picsum.photos/seed/ll/300/450"},
        {"title": "Whiplash",                "genre": "Drama,Music",             "year": 2014, "description": "A drummer pushes himself to perfection.",         "poster": "https://picsum.photos/seed/wh/300/450"},
        {"title": "The Social Network",      "genre": "Biography,Drama",         "year": 2010, "description": "The founding of Facebook.",                       "poster": "https://picsum.photos/seed/tsn/300/450"},
        {"title": "Arrival",                 "genre": "Drama,Mystery,Sci-Fi",    "year": 2016, "description": "A linguist communicates with aliens.",             "poster": "https://picsum.photos/seed/arr/300/450"},
        {"title": "Blade Runner 2049",       "genre": "Action,Drama,Mystery",    "year": 2017, "description": "A blade runner uncovers a secret.",               "poster": "https://picsum.photos/seed/br/300/450"},
        {"title": "1917",                    "genre": "Drama,War",               "year": 2019, "description": "Two soldiers race to deliver a message.",         "poster": "https://picsum.photos/seed/1917/300/450"},
        {"title": "Spirited Away",           "genre": "Animation,Adventure",     "year": 2001, "description": "A girl enters a spirit world.",                   "poster": "https://picsum.photos/seed/sa/300/450"},
        {"title": "Joker",                   "genre": "Crime,Drama,Thriller",    "year": 2019, "description": "The origin story of the Joker.",                 "poster": "https://picsum.photos/seed/jok/300/450"},
    ]

    for m in sample_movies:
        content = Content(**m,
                          avg_rating=round(random.uniform(3.5, 5.0), 1),
                          rating_count=random.randint(50, 500))
        db.session.add(content)

    demo = User(
        username="demo",
        email="demo@example.com",
        password_hash=bcrypt.generate_password_hash("password123").decode("utf-8"),
        preferred_genres="Action,Drama,Sci-Fi",
        is_admin=True
    )
    db.session.add(demo)
    db.session.commit()

    movies = Content.query.all()
    for movie in movies[:10]:
        rating = Rating(user_id=demo.id, content_id=movie.id,
                        rating=random.uniform(1, 5))
        db.session.add(rating)
    db.session.commit()
    print("✅ Sample data seeded successfully.")


# ── This block only runs locally, NOT on Render ──────────────
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
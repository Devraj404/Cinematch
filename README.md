# 🎬 CineMatch – Content Recommendation System

> **College Major Project** | ML-powered movie recommendation engine demonstrating Content-Based Filtering, Collaborative Filtering, and a Hybrid approach.

---

## 📋 Table of Contents
1. [Project Overview](#overview)
2. [System Architecture](#architecture)
3. [Technology Stack](#stack)
4. [ML Algorithms Explained](#algorithms)
5. [Project Structure](#structure)
6. [Setup & Installation](#setup)
7. [API Reference](#api)
8. [Dataset](#dataset)
9. [Evaluation Metrics](#metrics)
10. [Deployment](#deployment)

---

## Overview

CineMatch recommends movies to users based on:
- **Their own taste** (content-based: what genres/styles they like)
- **Community wisdom** (collaborative: what similar users love)
- **Both combined** (hybrid: the best of both worlds)

This mirrors how Netflix, Amazon, and YouTube build their recommendation systems.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                           │
│   Browser (Login · Dashboard · Browse · Recommendations)   │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP / REST API
┌─────────────────────────▼───────────────────────────────────┐
│                    APPLICATION LAYER                        │
│              Flask Backend (Python)                         │
│   ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌────────────┐  │
│   │ Auth API │ │Content API│ │Recommend  │ │ Admin API  │  │
│   │ /api/auth│ │/api/content│ │  API     │ │/api/admin  │  │
│   └──────────┘ └──────────┘ └─────┬─────┘ └────────────┘  │
└──────────────────────────────────┬─┴───────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────┐
│               RECOMMENDATION ENGINE LAYER                    │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Content-Based  │  │  Collaborative  │  │   Hybrid    │ │
│  │  TF-IDF +       │  │  User-Item      │  │  α·CF +     │ │
│  │  Cosine Sim     │  │  Matrix + KNN   │  │  (1-α)·CB   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│                   scikit-learn · pandas · numpy              │
└──────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────┐
│                      DATABASE LAYER                          │
│                SQLite (via SQLAlchemy ORM)                   │
│   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐   │
│   │  Users   │   │  Content │   │       Ratings        │   │
│   │ Table    │   │  Table   │   │  (user_id,content_id) │   │
│   └──────────┘   └──────────┘   └──────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer            | Technology                        |
|------------------|-----------------------------------|
| Backend          | Python 3.10+, Flask 3.0           |
| Authentication   | JWT (flask-jwt-extended)          |
| Database         | SQLite + SQLAlchemy ORM           |
| ML               | scikit-learn, pandas, numpy       |
| Visualization    | matplotlib, plotly                |
| Frontend         | Vanilla HTML/CSS/JavaScript       |
| Fonts            | Google Fonts (Bebas Neue, DM Sans)|

---

## ML Algorithms Explained

### 1. Content-Based Filtering
```
User likes: "Inception" (Action, Sci-Fi, 2010)
         ↓
Build "soup": "Action Sci-Fi Action Sci-Fi 2010"
         ↓
TF-IDF Vector: [0.3, 0.0, 0.5, 0.2, ...]
         ↓
Cosine Similarity with all movies
         ↓
Top-N most similar: "The Matrix", "Interstellar", ...
```

**Pros:** Works for new users (only needs their genre preferences)  
**Cons:** Limited to item features; can't discover outside stated preferences

### 2. Collaborative Filtering
```
User 1 ratings: [5, 4, 0, 3, 5, 0]
User 2 ratings: [4, 5, 0, 3, 4, 0]  ← most similar to User 1
User 3 ratings: [0, 0, 5, 1, 0, 4]  ← least similar
         ↓
User 2 liked movie #6 with rating 4 → recommend to User 1
```

**Pros:** Discovers unexpected gems; leverages community wisdom  
**Cons:** Cold start problem for new users/items; needs sufficient ratings

### 3. Hybrid System
```
CB_score  = normalize(content_based_score)  ∈ [0, 1]
CF_score  = normalize(collaborative_score)  ∈ [0, 1]

α = 0.6  (favor CF slightly)

final_score = α × CF_score + (1-α) × CB_score

If user has < 3 ratings → fall back to trending (cold start)
```

**Pros:** Best accuracy; handles cold start  
**Cons:** More complex to implement and tune

---

## Project Structure

```
recommendation-system/
│
├── app.py                    # Flask app factory + seed data
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── backend/
│   ├── models.py             # SQLAlchemy ORM models (User, Content, Rating)
│   └── routes/
│       ├── auth.py           # /api/auth/* (register, login, profile)
│       ├── content.py        # /api/content/* (browse, rate, search)
│       ├── recommend.py      # /api/recommend/* (for-me, similar, evaluate)
│       └── admin.py          # /api/admin/* (add content, stats)
│
├── models/
│   └── recommendation_engine.py  # All 3 ML algorithms + evaluator
│
├── frontend/
│   └── templates/
│       ├── index.html        # Login / Register page
│       ├── dashboard.html    # Main app (browse, trending, recs)
│       └── admin.html        # Admin panel
│
├── utils/
│   └── dataset_loader.py    # MovieLens loader + preprocessor
│
├── notebooks/
│   └── ml_exploration.ipynb # EDA, algorithm demos, visualizations
│
├── dataset/
│   └── ml-100k/             # Place MovieLens files here (optional)
│
└── docs/
    ├── similarity_heatmap.png
    ├── eda_charts.png
    └── metrics_chart.png
```

---

## Setup & Installation

```bash
# 1. Clone / download the project
cd recommendation-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py

# 5. Open in browser
# → http://localhost:5000
# Demo login: demo@example.com / password123
```

---

## API Reference

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create new account |
| POST | `/api/auth/login` | Login, get JWT token |
| GET | `/api/auth/profile` | Get current user profile |
| PUT | `/api/auth/profile` | Update genre preferences |

### Content
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/content/` | Browse movies (paginated, filterable) |
| GET | `/api/content/trending` | Top-rated movies |
| GET | `/api/content/<id>` | Movie details |
| POST | `/api/content/<id>/rate` | Submit rating (1–5) |
| GET | `/api/content/genres` | All genres |

### Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/recommend/for-me?algo=hybrid` | Personalized recs |
| GET | `/api/recommend/similar/<id>` | Similar movies |
| POST | `/api/recommend/retrain` | Retrain ML model |
| GET | `/api/recommend/evaluate` | Run evaluation metrics |

### Admin (requires admin account)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/content` | Add new movie |
| DELETE | `/api/admin/content/<id>` | Delete movie |
| GET | `/api/admin/stats` | System statistics |
| GET | `/api/admin/users` | All users |

---

## Dataset

### Option A: Built-in Sample Data (default)
The app auto-seeds 20 movies on first run. No setup needed.

### Option B: MovieLens 100K Dataset
1. Download from https://grouplens.org/datasets/movielens/100k/
2. Extract to `dataset/ml-100k/`
3. In `app.py`, replace `_seed_sample_data()` with `MovieLensLoader().load()`

**MovieLens Stats:**
- 100,000 ratings (1–5)
- 943 users
- 1,682 movies
- Sparsity: ~93.7%

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision@K** | hits@K / K | % of top-K recs that are relevant |
| **Recall@K** | hits@K / relevant | % of relevant items found in top-K |
| **F1@K** | 2·P·R / (P+R) | Harmonic mean of Precision & Recall |
| **RMSE** | √(Σ(actual-pred)²/n) | Rating prediction accuracy |

---

## Deployment

### Local (default)
```bash
python app.py  # Runs on http://localhost:5000
```

### Render.com (free tier)
```yaml
# render.yaml
services:
  - type: web
    name: cinematch
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:create_app()
    envVars:
      - key: SECRET_KEY
        generateValue: true
```

### Railway / Heroku
```
# Procfile
web: gunicorn "app:create_app()" --bind 0.0.0.0:$PORT
```

### AWS EC2
```bash
# Install dependencies, set up nginx as reverse proxy
# Use gunicorn for production WSGI server
gunicorn -w 4 -b 0.0.0.0:8000 "app:create_app()"
```

---

## Bonus: Advanced Features

| Feature | How to Implement |
|---------|-----------------|
| **Deep Learning** | Replace TF-IDF with BERT embeddings (sentence-transformers) |
| **Real-time recs** | WebSockets + retrain on each rating event |
| **User behavior** | Log page views, click-through rates, watch time |
| **Cold start** | Use demographic data + genre preferences for new users |
| **A/B testing** | Randomly serve CB vs CF, measure click-through |
| **Matrix Factorization** | SVD or ALS for better CF performance |

---

*Built for educational purposes. Demonstrates production-grade ML patterns in a college project context.*

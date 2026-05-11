"""
Routes: React app serving + retrieval API.

To enable LLM endpoints (chat + enrichments), set USE_LLM = True below
and add SPARK_API_KEY to .env. See llm_routes.py for LLM code.
"""
import json
import os
from flask import send_from_directory, request, jsonify
from models import db, Episode, Review
from retrieval import search as retrieval_search
from retrieval import search_programs as retrieval_search_programs
from retrieval import search_reddit as retrieval_search_reddit

# ── AI toggle ────────────────────────────────────────────────────────────────
#USE_LLM = False
USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


def json_search(query):
    if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return matches


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.get_json(force=True)
        query = data.get("query", "")
        if not query.strip():
            return jsonify({"results": []})

        raw_equipment = data.get("equipment")
        equipment = [e for e in raw_equipment if isinstance(e, str)] if isinstance(raw_equipment, list) else None
        if equipment == []:
            equipment = None

        raw_difficulty = data.get("difficulty")
        max_level = raw_difficulty if isinstance(raw_difficulty, str) and raw_difficulty else None

        raw_injuries = data.get("injuries")
        injured_muscles = [m for m in raw_injuries if isinstance(m, str)] if isinstance(raw_injuries, list) else None
        if injured_muscles == []:
            injured_muscles = None

        raw_method = data.get("method")
        method = raw_method if raw_method in ("tfidf", "svd") else "tfidf"

        payload = retrieval_search(
            query,
            k=20,
            equipment=equipment,
            max_level=max_level,
            injured_muscles=injured_muscles,
            method=method,
        )
        return jsonify(payload)

    @app.route("/api/search_programs", methods=["POST"])
    def search_programs_route():
        data = request.get_json(force=True)
        query = data.get("query", "")
        if not query.strip():
            return jsonify({"results": [], "corrected_query": None})
        raw_method = data.get("method")
        method = raw_method if raw_method in ("tfidf", "svd") else "tfidf"
        return jsonify(retrieval_search_programs(query, k=20, method=method))

    @app.route("/api/search_reddit", methods=["POST"])
    def search_reddit_route():
        data = request.get_json(force=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"results": [], "corrected_query": None})
        raw_method = data.get("method")
        method = raw_method if raw_method in ("tfidf", "svd") else "tfidf"
        raw_sports = data.get("sports")
        sports = [s for s in raw_sports if isinstance(s, str)] if isinstance(raw_sports, list) else None
        if sports == []:
            sports = None
        return jsonify(retrieval_search_reddit(query, k=20, method=method, sports=sports))

    if USE_LLM:
        from llm_routes import register_chat_route, register_enrichment_routes
        from rag_routes import register_rag_routes
        register_chat_route(app, json_search)
        register_enrichment_routes(app)
        register_rag_routes(app)

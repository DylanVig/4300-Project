"""
LLM routes — only loaded when USE_LLM = True in routes.py.

Endpoints:
  POST /api/chat              — SSE-streaming RAG chat (Kardashians demo).
  POST /api/enrich_exercise   — SSE-streaming complementary workout plan for the top exercise.
  POST /api/enrich_program    — JSON form cues + safety notes for a list of exercise names.

Setup:
  1. Add SPARK_API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient
from retrieval import search as retrieval_search

logger = logging.getLogger(__name__)

MAX_PROGRAM_EXERCISES = 40


def _get_llm_client():
    """Return an LLMClient or None if SPARK_API_KEY is unset."""
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        return None
    return LLMClient(api_key=api_key)


def _extract_json(text):
    """Pull the first JSON object out of an LLM response (handles ```json fences)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Keeping Up with the Kardashians episode titles, "
                "descriptions, and IMDB ratings. Search is by a single word in the episode title. "
                "Reply with exactly: YES followed by one space and ONE word to search (e.g. YES wedding), "
                "or NO if the question does not need episode data."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(\w+)", content)
    if yes_match:
        return True, yes_match.group(1).lower()
    if re.search(r"\bYES\b", content):
        return True, "Kardashian"
    return False, None


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        client = _get_llm_client()
        if client is None:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            episodes = json_search(search_term or "Kardashian")
            context_text = "\n\n---\n\n".join(
                f"Title: {ep['title']}\nDescription: {ep['descr']}\nIMDB Rating: {ep['imdb_rating']}"
                for ep in episodes
            ) or "No matching episodes found."
            messages = [
                {"role": "system", "content": "Answer questions about Keeping Up with the Kardashians using only the episode information provided."},
                {"role": "user", "content": f"Episode information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Keeping Up with the Kardashians questions."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def register_enrichment_routes(app):
    """Register the three LLM enrichment endpoints. Called from routes.py."""

    @app.route("/api/enrich_exercise", methods=["POST"])
    def enrich_exercise():
        data = request.get_json() or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        primary = data.get("primaryMuscles") or []
        secondary = data.get("secondaryMuscles") or []
        equipment = data.get("equipment")
        instructions = data.get("instructions") or []

        client = _get_llm_client()
        if client is None:
            return jsonify({"error": "SPARK_API_KEY not set — add it to your .env file"}), 500

        exercise_block = (
            f"Exercise name: {name}\n"
            f"Primary muscles: {', '.join(primary) if primary else 'unspecified'}\n"
            f"Secondary muscles: {', '.join(secondary) if secondary else 'unspecified'}\n"
            f"Equipment: {equipment or 'unspecified'}\n"
            f"Instructions:\n" + "\n".join(f"- {step}" for step in instructions[:8])
        )

        # Use the exercises currently shown on the user's viewport as the candidate pool.
        # Fall back to IR retrieval only if the frontend sent nothing.
        raw_pool = data.get("pool") or []
        if raw_pool:
            candidates = [
                r for r in raw_pool
                if isinstance(r, dict) and r.get("name", "").strip().lower() != name.strip().lower()
            ]
        else:
            muscles = list(primary) + list(secondary)
            seed_query = " ".join(muscles) if muscles else name
            equip_filter = [equipment] if equipment and equipment != "other" else None
            ir_payload = retrieval_search(seed_query, k=20, equipment=equip_filter)
            candidates = [
                r for r in (ir_payload.get("results") or [])
                if r.get("name", "").strip().lower() != name.strip().lower()
            ]
        candidate_lines = "\n".join(
            f"- {r['name']} (muscles: {', '.join(r.get('primaryMuscles') or [])}; "
            f"equipment: {r.get('equipment') or 'unspecified'})"
            for r in candidates
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strength-and-conditioning coach. Given a centerpiece exercise and a "
                    "pool of available exercises retrieved from the database, output a full "
                    "complementary workout plan for the day. Include: a brief warmup, 2 accessory "
                    "exercises that target complementary muscle groups or movement patterns, "
                    "an optional finisher,and a cooldown. You MUST select all accessory exercises, and "
                    "the finisherexclusively from the provided exercise pool — do not invent exercises "
                    "not on the list. Match the available equipment. Keep tone direct and practical. "
                    "Use markdown headings and bullet lists. "
                    "For any exercise prescribed for hypertrophy, use a rep range of 8-12."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Centerpiece exercise:\n\n{exercise_block}\n\n"
                    f"Available exercise pool (choose from these only):\n{candidate_lines}\n\n"
                    "Write the workout plan."
                ),
            },
        ]

        def generate():
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"enrich_exercise streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/enrich_program", methods=["POST"])
    def enrich_program():
        data = request.get_json() or {}
        raw = data.get("exercises") or []
        seen = set()
        exercises = []
        for name in raw:
            if not isinstance(name, str):
                continue
            key = name.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            exercises.append(name.strip())
        exercises = exercises[:MAX_PROGRAM_EXERCISES]
        if not exercises:
            return jsonify({"cues": {}})

        client = _get_llm_client()
        if client is None:
            return jsonify({"cues": {}})

        listing = "\n".join(f"- {n}" for n in exercises)
        messages = [
            {
                "role": "system",
                "content": (
                    "You give concise strength-training form cues. Respond ONLY with valid JSON in "
                    "exactly this shape: "
                    "{\"cues\": {\"<exercise name>\": {\"form_cues\": [\"...\", \"...\"], \"safety\": \"...\"}}}. "
                    "Provide 2-3 short form cues (each under 15 words) and one safety note per exercise. "
                    "Use the exercise names exactly as given. No prose outside the JSON."
                ),
            },
            {
                "role": "user",
                "content": f"Exercises:\n{listing}",
            },
        ]

        try:
            response = client.chat(messages, stream=False)
            content = response.get("content") if isinstance(response, dict) else None
        except Exception as e:
            logger.error(f"enrich_program LLM error: {e}")
            return jsonify({"cues": {}})

        parsed = _extract_json(content) or {}
        cues_raw = parsed.get("cues") if isinstance(parsed, dict) else None
        cues = {}
        if isinstance(cues_raw, dict):
            for key, val in cues_raw.items():
                if not isinstance(key, str) or not isinstance(val, dict):
                    continue
                form_cues = val.get("form_cues") or []
                safety = val.get("safety") or ""
                if not isinstance(form_cues, list):
                    continue
                form_cues_clean = [c for c in form_cues if isinstance(c, str) and c.strip()]
                cues[key.strip()] = {
                    "form_cues": form_cues_clean,
                    "safety": safety.strip() if isinstance(safety, str) else "",
                }
        return jsonify({"cues": cues})

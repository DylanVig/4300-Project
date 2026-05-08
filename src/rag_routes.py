"""
RAG routes — query refinement + LLM reranking on top of the IR pipeline.

Only loaded when USE_LLM = True in routes.py. Separate from llm_routes.py
to keep enrichment logic (workout plans, form cues) disjoint from
retrieval-affecting RAG logic.

Flow per request:
    1. LLM rewrites the original query into a keyword-rich refined query.
    2. IR retrieves top-20 candidates using the *refined* query and the
       currently-selected method (tfidf | svd) plus any filters.
    3. LLM reranks those 20 down to top-5.
    4. Response carries results + the refined query so the UI can show
       the IR -> IR+RAG delta.

Endpoints:
    POST /api/rag_search            — exercises
    POST /api/rag_search_programs   — programs
"""
import json
import logging
from flask import request, jsonify

from llm_routes import _get_llm_client, _extract_json
from retrieval import search as retrieval_search
from retrieval import search_programs as retrieval_search_programs

logger = logging.getLogger(__name__)

CANDIDATE_POOL = 20
FINAL_K = 5


def _refine_query(client, original_query, domain):
    """Ask the LLM for a keyword-rich rewrite of the user's query.

    domain: "exercise" or "program" — selects vocabulary guidance.
    Falls back to the original query on any failure.
    """
    if domain == "program":
        vocab_hint = (
            "program goals (hypertrophy, powerlifting, bodybuilding, strength, "
            "endurance, cutting, bulking), periodization terms, split type "
            "(push/pull/legs, upper/lower, full body), experience level"
        )
    else:
        vocab_hint = (
            "specific muscle groups (quadriceps, hamstrings, glutes, pectorals, "
            "latissimus, deltoids, biceps, triceps, core), equipment names "
            "(barbell, dumbbell, kettlebell, cable, body only), movement "
            "patterns (squat, hinge, push, pull, carry), training qualities "
            "(hypertrophy, strength, power, plyometric, mobility)"
        )
    system = (
        "You rewrite fitness-search queries for a TF-IDF retrieval index. "
        "Given a user's natural-language query, output a single line of "
        "space-separated keywords that best describe what should be "
        "retrieved. Prefer canonical domain vocabulary: " + vocab_hint + ". "
        "No prose, no punctuation, no stop-words, no quoting. Just the "
        "keywords. 6-14 terms is ideal."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": original_query},
    ]
    try:
        response = client.chat(messages, stream=False)
        content = response.get("content") if isinstance(response, dict) else None
    except Exception as e:
        logger.warning(f"refine_query failed: {e}")
        return original_query
    if not content:
        return original_query
    cleaned = content.strip().strip('"').strip("'")
    # Keep a single line only.
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    return cleaned or original_query


def _exercise_candidate_line(idx, ex):
    primary = ", ".join(ex.get("primaryMuscles") or []) or "-"
    secondary = ", ".join(ex.get("secondaryMuscles") or []) or "-"
    return (
        f"{idx}. {ex.get('name')} | primary: {primary} | secondary: {secondary} "
        f"| equipment: {ex.get('equipment') or '-'} | level: {ex.get('level') or '-'} "
        f"| category: {ex.get('category') or '-'}"
    )


def _program_candidate_line(idx, pg):
    goals = ", ".join(pg.get("goal") or []) or "-"
    schedule = pg.get("schedule") or []
    first_exs = []
    seen = set()
    for entry in schedule:
        nm = (entry.get("exercise_name") or "").strip()
        key = nm.lower()
        if not nm or key in seen:
            continue
        seen.add(key)
        first_exs.append(nm)
        if len(first_exs) >= 5:
            break
    ex_preview = ", ".join(first_exs) or "-"
    weeks = pg.get("program_length_weeks")
    return (
        f"{idx}. {pg.get('title')} | goal: {goals} | level: {pg.get('level') or '-'} "
        f"| weeks: {weeks if weeks is not None else '-'} | first exercises: {ex_preview}"
    )


def _rerank_candidates(client, original_query, refined_query, candidates,
                       line_builder, k=FINAL_K):
    """Ask the LLM to rerank the candidate list down to top-k.

    Returns a list of indices into `candidates` (length <= k). On malformed
    output, returns list(range(min(k, len(candidates)))) as a safe fallback.
    """
    if not candidates:
        return []
    lines = [line_builder(i + 1, c) for i, c in enumerate(candidates)]
    listing = "\n".join(lines)
    system = (
        "You rerank fitness-retrieval candidates for relevance. Given a "
        "user's original query, a refined keyword query, and a numbered "
        f"candidate list, pick the {k} most relevant candidates in order "
        "of best fit. Consider the user's intent, not just keyword overlap. "
        "Respond ONLY with valid JSON of this exact shape: "
        "{\"ranked\": [n, n, ...]} where each n is a 1-based number from "
        "the candidate list. No prose."
    )
    user = (
        f"Original query: {original_query}\n"
        f"Refined query: {refined_query}\n\n"
        f"Candidates:\n{listing}\n\n"
        f"Return the top {k} as JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    fallback = list(range(min(k, len(candidates))))
    try:
        response = client.chat(messages, stream=False)
        content = response.get("content") if isinstance(response, dict) else None
    except Exception as e:
        logger.warning(f"rerank_candidates LLM error: {e}")
        return fallback
    parsed = _extract_json(content) or {}
    ranked_raw = parsed.get("ranked") if isinstance(parsed, dict) else None
    if not isinstance(ranked_raw, list):
        return fallback
    seen = set()
    picked = []
    for n in ranked_raw:
        try:
            idx = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            picked.append(idx)
        if len(picked) >= k:
            break
    # Backfill from IR order if the LLM gave us fewer than k valid picks.
    if len(picked) < k:
        for i in range(len(candidates)):
            if i not in seen:
                picked.append(i)
                seen.add(i)
                if len(picked) >= k:
                    break
    return picked[:k]


def _summarize_picks(client, original_query, refined_query, picks, line_builder, domain="exercise"):
    """Generate a 2-3 sentence prose summary helping the user pick from top-k results.

    Returns empty string on any failure so the UI can hide the block gracefully.
    """
    if not picks:
        return ""
    domain_label = "programs" if domain == "program" else "exercises"
    lines = [line_builder(i + 1, p) for i, p in enumerate(picks)]
    listing = "\n".join(lines)
    name_field = "title" if domain == "program" else "name"
    system = (
        f"You write a 2-3 sentence summary helping a user pick from a ranked "
        f"list of fitness {domain_label}. State which top result fits best and "
        "why, then briefly note 1-2 alternatives and what they would be "
        "preferred for. "
        f"You MUST refer to each {domain_label[:-1]} by its full {name_field} "
        "verbatim (exactly as written in the list, including capitalization "
        "and punctuation) — never use phrases like 'the first result', "
        "'the second option', or numeric ordinals. "
        "Plain prose, no markdown, no lists, no quotes."
    )
    user = (
        f"Original query: {original_query}\n"
        f"Refined query: {refined_query}\n\n"
        f"Top results:\n{listing}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = client.chat(messages, stream=False)
        content = response.get("content") if isinstance(response, dict) else None
        return (content or "").strip()
    except Exception as e:
        logger.warning(f"summarize_picks failed: {e}")
        return ""


def register_rag_routes(app):
    """Register the two RAG endpoints. Called from routes.py."""

    @app.route("/api/rag_search", methods=["POST"])
    def rag_search():
        data = request.get_json(force=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({
                "results": [], "corrected_query": None,
                "refined_query": "", "original_query": "",
            })

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

        client = _get_llm_client()
        if client is None:
            return jsonify({"error": "LLM not configured"}), 503

        refined = _refine_query(client, query, domain="exercise")

        ir_payload = retrieval_search(
            refined,
            k=CANDIDATE_POOL,
            equipment=equipment,
            max_level=max_level,
            injured_muscles=injured_muscles,
            method=method,
        )
        candidates = ir_payload.get("results") or []

        picks = _rerank_candidates(client, query, refined, candidates,
                                   _exercise_candidate_line)
        final = [candidates[i] for i in picks]
        summary = _summarize_picks(client, query, refined, final,
                                   _exercise_candidate_line, domain="exercise")

        return jsonify({
            "results": final,
            "corrected_query": ir_payload.get("corrected_query"),
            "refined_query": refined,
            "original_query": query,
            "summary": summary,
        })

    @app.route("/api/rag_search_programs", methods=["POST"])
    def rag_search_programs():
        data = request.get_json(force=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({
                "results": [], "corrected_query": None,
                "refined_query": "", "original_query": "",
            })
        raw_method = data.get("method")
        method = raw_method if raw_method in ("tfidf", "svd") else "tfidf"

        client = _get_llm_client()
        if client is None:
            return jsonify({"error": "LLM not configured"}), 503

        refined = _refine_query(client, query, domain="program")

        ir_payload = retrieval_search_programs(refined, k=CANDIDATE_POOL, method=method)
        candidates = ir_payload.get("results") or []

        picks = _rerank_candidates(client, query, refined, candidates,
                                   _program_candidate_line)
        final = [candidates[i] for i in picks]
        summary = _summarize_picks(client, query, refined, final,
                                   _program_candidate_line, domain="program")

        return jsonify({
            "results": final,
            "corrected_query": ir_payload.get("corrected_query"),
            "refined_query": refined,
            "original_query": query,
            "summary": summary,
        })

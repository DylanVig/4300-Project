"""
Retrieval module for the Athletic Training Finder.

Uses a Vector Space Model with TF-IDF weighting and cosine similarity to rank
exercises from the Free Exercise DB (873 exercises) against a user's
natural-language query.

Key IR techniques:
    - TF-IDF vectorization (scikit-learn TfidfVectorizer) over a combined text
      representation of each exercise.
    - Cosine similarity between the query vector and every exercise vector to
      produce a relevance score.
    - Field-priority weighting: higher-priority fields (e.g. primaryMuscles)
      are repeated more often in the document string so that matching terms in
      those fields contribute more to the TF-IDF score.
    - Query expansion: goal keywords like "vertical jump" are expanded into the
      muscle groups they target (quadriceps, glutes, calves, hamstrings), and
      equipment synonyms like "bodyweight" are mapped to dataset values
      ("body only") so the user doesn't need to know exact field terminology.
    - Spell correction: each query token is checked against the TF-IDF
      vocabulary using the Wagner-Fisher minimum edit distance algorithm.
      Tokens not found in the vocabulary are replaced with the closest
      vocabulary term within a maximum edit distance of 2.
    - Light load-time cleaning of the Free Exercise DB: secondaryMuscles is
      deduped against primaryMuscles so a muscle listed in both fields is
      not double-weighted. See data/DATASETS.md for the full data audit.
"""
import csv
import json
import os
import re
import sys
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Latent dimensions for TruncatedSVD (LSA) on top of the TF-IDF matrix.
# Both searchers use this as their upper bound; actual n_components is
# clamped to min(n_samples, n_features) - 1 for safety.
SVD_COMPONENTS = 100

# programs_cleaned.csv has very long schedule_json / exercise-list fields;
# bump the csv field size limit so the stdlib reader doesn't choke on them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ── Porter Stemmer ───────────────────
# Implements Martin Porter's algorithm (1980).  Reduces inflected/derived forms
# to a common stem so that "strengthening", "strengthens", and "strength" all
# map to the same index term.
#
# The algorithm is split into six steps, each applying suffix-replacement rules
# gated on the "measure" m — the number of consonant-vowel (VC) sequences in
# the candidate stem.  A higher measure means a longer, less ambiguous stem and
# therefore a more aggressive replacement is safe.

_VOWELS = frozenset('aeiou')

# Common English stop words handled in the tokenizer so we can pass
# stop_words=None to TfidfVectorizer and avoid sklearn's "inconsistent
# stop-words" warning when a custom tokenizer is used.
_STOP_WORDS = frozenset([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'its', 'be', 'are', 'was',
    'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'as', 'this',
    'that', 'these', 'those', 'which', 'who', 'what', 'how', 'when', 'where',
    'while', 'if', 'not', 'no', 'so', 'than', 'then', 'also', 'more', 'most',
    'other', 'such', 'same', 'your', 'you', 'they', 'their', 'them', 'he',
    'she', 'his', 'her', 'we', 'our', 'us', 'me', 'my', 'i', 'each', 'both',
    # Domain-specific stop words: too generic to discriminate in an exercise DB
    'exercise', 'exercises', 'workout', 'workouts', 'target', 'targets',
    'targeting', 'muscle', 'muscles', 'movement', 'movements', 'position',
    'body', 'starting', 'repeat',
])


def _is_consonant(word, i):
    """Return True if word[i] is a consonant.

    'y' is treated as a consonant when it immediately follows another consonant
    (e.g. the 'y' in 'sky'), and as a vowel otherwise (e.g. 'yes').
    """
    c = word[i]
    if c in _VOWELS:
        return False
    if c == 'y':
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(word):
    """Count VC sequences (Porter's measure m) in *word*.

    The measure m counts how many times a vowel run is followed by a consonant
    run: [C](VC)^m[V].  A stem with m > 0 has at least one interior vowel and
    is long enough for most suffix rules to apply safely.
    """
    n, i, m = len(word), 0, 0
    while i < n and _is_consonant(word, i):
        i += 1                              # skip leading consonant cluster
    while i < n:
        while i < n and not _is_consonant(word, i):
            i += 1                          # skip vowel run
        if i < n:
            m += 1                          # counted one VC pair
            while i < n and _is_consonant(word, i):
                i += 1                      # skip consonant run
    return m


def _has_vowel(word):
    """Return True if *word* contains at least one vowel."""
    return any(not _is_consonant(word, i) for i in range(len(word)))


def _ends_double_consonant(word):
    """Return True if *word* ends with the same consonant twice (e.g. 'tt')."""
    return (len(word) >= 2
            and word[-1] == word[-2]
            and _is_consonant(word, len(word) - 1))


def _ends_cvc(word):
    """Return True if *word* ends with consonant-vowel-consonant, last c ∉ {w,x,y}.

    Used in step 1b to decide whether to append 'e' after removing -ed/-ing.
    """
    n = len(word)
    return (n >= 3
            and _is_consonant(word, n - 1)
            and not _is_consonant(word, n - 2)
            and _is_consonant(word, n - 3)
            and word[-1] not in 'wxy')


def _step1ab(word):
    """Step 1a: handle plurals.  Step 1b: handle -ed and -ing."""
    # Step 1a
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif not word.endswith('ss') and word.endswith('s'):
        word = word[:-1]

    # Step 1b
    if word.endswith('eed'):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith('ed') or word.endswith('ing'):
        stem = word[:-2] if word.endswith('ed') else word[:-3]
        if _has_vowel(stem):
            word = stem
            if word.endswith(('at', 'bl', 'iz')):
                word += 'e'
            elif _ends_double_consonant(word) and word[-1] not in 'lsz':
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += 'e'
    return word


def _step1c(word):
    """Step 1c: replace trailing y with i when the stem has a vowel."""
    if word.endswith('y') and len(word) > 1 and _has_vowel(word[:-1]):
        word = word[:-1] + 'i'
    return word


def _step2(word):
    """Step 2: reduce double-suffixes (m > 0 required)."""
    rules = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
        ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
        ('alli', 'al'),  ('entli', 'ent'), ('eli', 'e'),
        ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
        ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
        ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
        ('iviti', 'ive'), ('biliti', 'ble'),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and _measure(word[:-len(suffix)]) > 0:
            return word[:-len(suffix)] + replacement
    return word


def _step3(word):
    """Step 3: reduce single suffixes (m > 0 required)."""
    rules = [
        ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
        ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', ''),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and _measure(word[:-len(suffix)]) > 0:
            return word[:-len(suffix)] + replacement
    return word


def _step4(word):
    """Step 4: strip derivational suffixes (m > 1 required)."""
    for suffix in ('al', 'ance', 'ence', 'er', 'ic', 'able', 'ible',
                   'ant', 'ement', 'ment', 'ent', 'ou', 'ism', 'ate',
                   'iti', 'ous', 'ive', 'ize'):
        if word.endswith(suffix) and _measure(word[:-len(suffix)]) > 1:
            return word[:-len(suffix)]
    # -ion is only removed when the preceding letter is s or t
    if word.endswith('ion'):
        stem = word[:-3]
        if _measure(stem) > 1 and stem and stem[-1] in 'st':
            return stem
    return word


def _step5(word):
    """Step 5: tidy up final e and double-l."""
    # Step 5a: remove trailing e
    if word.endswith('e'):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            word = stem
    # Step 5b: remove doubled l when m > 1
    if _ends_double_consonant(word) and word.endswith('l') and _measure(word[:-1]) > 1:
        word = word[:-1]
    return word


def _stem(word):
    """Reduce *word* to its Porter stem (pure Python, no libraries).

    Words of length ≤ 2 are returned unchanged — the algorithm is not
    reliable on very short strings and they're unlikely to have inflections
    worth stripping.
    """
    if len(word) <= 2:
        return word
    word = _step1ab(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5(word)
    return word


# Surface-form aliases applied BEFORE stemming, at both index and query time.
# These fix corpus-level inconsistencies and common slang that Porter stemming
# alone can't resolve:
#   - "dumbell" (one b) and "dumbbell" (two b's) both appear in the corpus and
#     stem to DIFFERENT terms (`dumbel` vs `dumbbel`), so queries for one miss
#     documents using the other. Canonicalizing to "dumbbell" merges them.
#   - "hammies" is common gym slang that isn't in the corpus; mapping it to
#     "hamstrings" routes it to a real vocab term.
# Keep this table SHORT: each entry affects indexing, so additions require
# rebuilding the searcher. Only add entries for real retrieval-quality bugs,
# not cosmetic ones.
_TOKEN_ALIASES = {
    "dumbell": "dumbbell",
    "dumbells": "dumbbells",
    "hammies": "hamstrings",
}


def _apply_alias(token):
    return _TOKEN_ALIASES.get(token, token)


def _tokenize_and_stem(text):
    """Tokenize *text* into lowercase alpha tokens, remove stop words, then stem.

    Used as the custom tokenizer for TfidfVectorizer so that index terms and
    query terms go through identical preprocessing.
    """
    tokens = re.findall(r'[a-z]+', text.lower())
    return [_stem(_apply_alias(t)) for t in tokens if t not in _STOP_WORDS and len(t) > 1]

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'exercises_free_db.json')
PROGRAMS_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'programs_cleaned.csv')
COMMON_ENGLISH_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   'google-10000-english-no-swears.txt')


def _load_english_words(path=COMMON_ENGLISH_PATH):
    """Load a frozenset of common English words for the spell-correction gate.

    The searchers use this to decide whether a query token is likely a typo
    (not English) or a legitimate word the user really typed. Without this
    gate, ``_correct_token`` rewrites any non-vocab token to the nearest
    corpus term within edit distance 2, which mangles common English words
    that simply don't appear in the exercise corpus (e.g. "bigger" → "finger",
    "build" → "bulk").

    A missing file falls back to an empty set so the module still imports in
    minimal environments — correction then behaves as it did before the gate
    was added.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return frozenset(line.strip().lower() for line in f if line.strip())
    except OSError:
        return frozenset()


_ENGLISH_WORDS = _load_english_words()

# ── Query expansion maps ─────────────────────────────────────────────────────
# GOAL_TO_MUSCLES maps common fitness goal keywords to the muscle groups they
# primarily involve. When any of these keywords appear in the user query, the
# corresponding muscle names are appended to the query string before TF-IDF
# vectorization.

GOAL_TO_MUSCLES = {
    "vertical jump": ["quadriceps", "glutes", "calves", "hamstrings"],
    "jump": ["quadriceps", "glutes", "calves"],
    "squat": ["quadriceps", "glutes", "hamstrings"],
    "posture": ["lower back", "abdominals", "shoulders"],
    "running": ["hamstrings", "calves", "abdominals"],
    "speed": ["hamstrings", "quadriceps", "calves"],
    "shooting": ["shoulders", "triceps", "forearms"],
    "throw": ["shoulders", "triceps", "chest"],
    "core": ["abdominals", "lower back"],
    "abs": ["abdominals"],
    "back": ["lats", "middle back", "lower back"],
    "chest": ["chest"],
    "arms": ["biceps", "triceps", "forearms"],
    "legs": ["quadriceps", "hamstrings", "calves", "glutes"],
    "shoulders": ["shoulders", "traps"],
    "grip": ["forearms"],
    "push": ["chest", "triceps", "shoulders"],
    "pull": ["lats", "biceps", "middle back"],
}

# EQUIPMENT_KEYWORDS maps user-friendly equipment terms to the exact values
# used in the dataset's "equipment" field.  A value of None (e.g. "gym") means
# "don't filter by equipment" — the user is saying all equipment is available.

EQUIPMENT_KEYWORDS = {
    "no equipment": "body only",
    "bodyweight": "body only",
    "gym": None,
    "dumbbells": "dumbbell",
    "barbell": "barbell",
    "cables": "cable",
    "machine": "machine",
    "bands": "bands",
    "kettlebell": "kettlebells",
}

# ── Field weights ─────────────────────────────────────────────────────────────
# Each exercise field is assigned a priority weight (1.0 = highest).  During
# document construction, each field's text is repeated proportionally to its
# weight (weight * 5, minimum 1 repetition).  This causes TF-IDF to assign
# higher term frequencies to tokens that appear in important fields, so that
# e.g. a query term matching primaryMuscles contributes more to the cosine
# similarity score than the same term matching the exercise name.

FIELD_WEIGHTS = {
    "primaryMuscles": 1.0,
    "secondaryMuscles": 0.4,
    "category": 0.6,
    "level": 0.5,
    "equipment": 0.4,
    "name": 0.3,
    "mechanic": 0.2,
    "force": 0.2,
}


def _safe(val):
    """Convert a field value to a lowercase string safe for concatenation.

    Args:
        val: A field value from the exercise JSON — may be a string, a list of
             strings, or None.

    Returns:
        A single space-joined string.  None becomes "", lists are joined with
        spaces, and scalar values are cast to str.
    """
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(val)
    return str(val)


def _dedup_secondary(ex):
    """Return secondaryMuscles with any entries that also appear in
    primaryMuscles removed.

    9 exercises in the upstream Free Exercise DB list the same muscle in both
    primaryMuscles and secondaryMuscles (e.g. "Barbell Step Ups" has
    quadriceps in both). Without this dedup, those muscles would be weighted
    1.0 (primary) + 0.4 (secondary) = 1.4x, giving those 9 docs a small but
    unearned TF bump on the shared muscle term. See data/DATASETS.md.
    """
    primary = set(ex.get("primaryMuscles") or [])
    secondary = ex.get("secondaryMuscles") or []
    return [m for m in secondary if m not in primary]


def _build_weighted_doc(ex):
    """Build a single TF-IDF document string for one exercise.

    Concatenates exercise fields into a single string, repeating each field's
    text proportionally to its priority weight so that the TF-IDF vectorizer
    naturally assigns higher term frequencies to high-priority fields.

    For example, with weight 1.0 and a multiplier of 5, the primaryMuscles
    text appears 5 times, while mechanic (weight 0.2) appears only once.
    Instructions are appended once at the end (unweighted) to provide
    additional context without dominating the score.

    secondaryMuscles is deduped against primaryMuscles (see _dedup_secondary)
    so that a muscle listed in both fields is not double-weighted.

    Args:
        ex: A single exercise dict from the Free Exercise DB JSON.

    Returns:
        A space-joined string ready for TF-IDF vectorization.
    """
    parts = []
    for field, weight in FIELD_WEIGHTS.items():
        if field == "secondaryMuscles":
            text = _safe(_dedup_secondary(ex))
        else:
            text = _safe(ex.get(field, ""))
        if text:
            repeat = max(1, int(weight * 5))  # scale weights into repeat counts
            parts.extend([text] * repeat)
    instructions = " ".join(ex.get("instructions", []))
    if instructions:
        parts.append(instructions)
    return " ".join(parts)


def _fit_svd(tfidf_matrix, n_components=SVD_COMPONENTS, random_state=42):
    """Fit a TruncatedSVD (LSA) over a TF-IDF matrix and row-normalize.

    Returns ``(svd, svd_matrix_normed)`` where ``svd`` is the fitted
    transformer (reused at query time to project a TF-IDF query vector
    into the same latent space) and ``svd_matrix_normed`` is the
    document-by-component matrix with each row scaled to unit L2 norm
    so that cosine similarity at query time reduces to a single dot
    product ``svd_matrix_normed @ q_normed.T``.

    The caller is expected to L2-normalize the query vector before the
    dot product — see ``_svd_scores``.

    ``n_components`` is clamped to ``min(tfidf_matrix.shape) - 1`` so the
    decomposition remains valid on small corpora or narrow vocabularies.
    """
    n_comp = min(n_components, min(tfidf_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    reduced = svd.fit_transform(tfidf_matrix)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return svd, reduced / norms


def _svd_scores(svd, svd_matrix_normed, query_vec):
    """Score every document against a TF-IDF query vector in latent space.

    Projects the (already TF-IDF transformed) query into the SVD space,
    L2-normalizes it, then returns the 1-D cosine similarity array
    against every row of ``svd_matrix_normed``. Because both sides are
    unit-normalized, the dot product equals cosine similarity.
    """
    reduced = svd.transform(query_vec)
    qnorm = np.linalg.norm(reduced)
    if qnorm > 0:
        reduced = reduced / qnorm
    return (svd_matrix_normed @ reduced.T).flatten()


# ── Result-explanation helpers ───────────────────
# Surface *why* a result matched (tags) and *how good* the match is
# relative to the full top-k (match_quality). Both are attached to
# every hit at search time so the frontend can explain the raw cosine
# score to the user.

N_TOPICS = 28          # NMF components used purely for theme labels
TAGS_PER_RESULT = 3

# Absolute quality thresholds per method. Tunable constants — see
# context/improvements.md if empirical calibration is ever run.
_QUALITY_BANDS = {
    "tfidf": (0.30, 0.15),  # (strong_min, moderate_min)
    "svd":   (0.50, 0.25),
}
_RELATIVE_BOOST_FRAC = 0.80  # ≥80% of top → bump one band


def _build_stem_surface_map(docs):
    """Map each stem to the most frequent raw surface form seen across *docs*.

    The TF-IDF vocabulary stores stems (``tricep``, ``explos``, ``pector``)
    which read as garbled fragments to a user. This scan passes through the
    same tokenization + aliasing as indexing but skips stemming, counts
    each ``(stem, surface)`` pair's occurrences, and picks the most common
    surface for each stem. Used to un-stem tags before display.

    Returns ``{stem: surface_form}``. Stems not seen in *docs* won't have
    an entry, so callers should fall back to the stem itself.
    """
    from collections import Counter
    counts = {}
    for doc in docs:
        tokens = re.findall(r'[a-z]+', doc.lower())
        for raw in tokens:
            if raw in _STOP_WORDS or len(raw) <= 1:
                continue
            aliased = _apply_alias(raw)
            stem = _stem(aliased)
            if stem not in counts:
                counts[stem] = Counter()
            counts[stem][aliased] += 1
    return {stem: c.most_common(1)[0][0] for stem, c in counts.items()}


def _unstem(term, stem_to_surface):
    """Look up a display form for *term*, falling back to *term* itself."""
    return stem_to_surface.get(term, term) if stem_to_surface else term


def _top_shared_terms(vectorizer, query_vec, doc_vec, stem_to_surface=None,
                     k=TAGS_PER_RESULT):
    """Top-k features driving the TF-IDF match between a query and one doc.

    Element-wise product of the two sparse rows recovers each term's
    contribution to the (un-normalized) cosine dot product; the largest
    values name the terms most responsible for the match. Stems are
    mapped back to their most frequent surface form via *stem_to_surface*
    when provided.
    """
    contrib = query_vec.multiply(doc_vec).tocsr()
    if contrib.nnz == 0:
        return []
    row = contrib.getrow(0)
    indices = row.indices
    data = row.data
    if data.size == 0:
        return []
    order = np.argsort(data)[::-1][:k]
    feats = vectorizer.get_feature_names_out()
    return [_unstem(str(feats[indices[i]]), stem_to_surface) for i in order]


def _fit_topic_nmf(tfidf_matrix, vectorizer, stem_to_surface,
                   n_topics=N_TOPICS, random_state=42, required_labels=None):
    """Fit a small NMF over the TF-IDF matrix for per-result topic labels.

    NMF components are non-negative and additive, so each component
    captures a cluster of co-occurring terms (a topic) rather than an
    orthogonal contrast axis. Each topic gets a single-term label: the
    highest IDF-weighted term in the component, deduped across topics so
    each visible concept appears on at most one chip.

    ``required_labels`` is an optional list of concept strings (e.g. muscle
    group names, program goal categories) that must appear in the final label
    set. After the initial IDF-weighted pass, any required label not yet
    covered is assigned to the component that loads most heavily on that
    concept's most-discriminative vocabulary token, overriding whatever label
    that component received in the first pass. This guarantees coverage of
    important concepts that IDF suppresses (e.g. "back" is so common its IDF
    is low, so it loses to "lats" in an unguided ranking).
    """
    from sklearn.decomposition import NMF
    n_comp = max(1, min(n_topics, min(tfidf_matrix.shape) - 1))
    nmf = NMF(n_components=n_comp, init="nndsvda",
              random_state=random_state, max_iter=400)
    doc_topics = nmf.fit_transform(tfidf_matrix)
    feats = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_

    labels, used = [], set()
    for topic_idx in range(n_comp):
        weighted = nmf.components_[topic_idx] * idf
        for i in np.argsort(weighted)[::-1]:
            term = _unstem(str(feats[i]), stem_to_surface)
            if term not in used:
                labels.append(term)
                used.add(term)
                break
        else:
            term = _unstem(str(feats[np.argmax(weighted)]), stem_to_surface)
            labels.append(term)
            used.add(term)

    if required_labels:
        feat_to_idx = {str(f): i for i, f in enumerate(feats)}
        required_set = {r.lower().strip() for r in required_labels}
        for req in required_labels:
            req_display = req.lower().strip()
            if req_display in used:
                continue
            tokens = [_stem(_apply_alias(t))
                      for t in re.findall(r'[a-z]+', req_display)]
            tokens = [t for t in tokens
                      if t in feat_to_idx and t not in _STOP_WORDS and len(t) > 1]
            if not tokens:
                continue
            # Use the most discriminative vocab token (highest IDF) to find
            # which NMF component best represents this concept.
            key = max(tokens, key=lambda t: idf[feat_to_idx[t]])
            feat_idx = feat_to_idx[key]
            loadings = nmf.components_[:, feat_idx]
            # Prefer a component whose current label is not already a required
            # label — that avoids displacing sibling concepts that were already
            # correctly assigned (e.g. "middle back" shouldn't steal "lats").
            best_comp = None
            for c in np.argsort(loadings)[::-1]:
                if loadings[c] == 0:
                    break
                if labels[c] not in required_set:
                    best_comp = int(c)
                    break
            if best_comp is None:
                continue  # no spare component available; skip rather than displace
            old = labels[best_comp]
            used.discard(old)
            labels[best_comp] = req_display
            used.add(req_display)

    baseline = np.maximum(doc_topics.mean(axis=0), 1e-6)
    return nmf, doc_topics, labels, baseline


def _top_topics(query_topic_vec, doc_topic_vec, baseline=None,
                k=TAGS_PER_RESULT, query_weak_ratio=0.15):
    """Top-k NMF topic indices explaining why ``doc_topic_vec`` matched the query.

    NMF loadings are non-negative so there is no sign branch. Contribution
    per topic = ``q_t * doc_t``, optionally divided by per-topic corpus
    mean for distinctiveness. Topics where the query loads weakly (below
    ``query_weak_ratio × max``) are masked out first.
    """
    q_max = float(query_topic_vec.max()) if query_topic_vec.size else 0.0
    if q_max == 0.0:
        return []
    mask = query_topic_vec >= query_weak_ratio * q_max
    contrib = query_topic_vec * doc_topic_vec
    if baseline is not None:
        contrib = contrib / baseline
    contrib = np.where(mask, contrib, -np.inf)
    if not np.any(contrib > 0):
        return []
    order = np.argsort(contrib)[::-1][:k]
    return [int(t) for t in order if contrib[t] > 0]


def _match_quality(score, top_score, method):
    """Bucket a raw cosine score into 'strong' / 'moderate' / 'weak'.

    Uses absolute bands per method, then bumps up one band if ``score``
    is within ``_RELATIVE_BOOST_FRAC`` of ``top_score`` so the user's
    best option always reads as meaningful even when the query as a
    whole is weak.
    """
    strong_min, mod_min = _QUALITY_BANDS.get(method, _QUALITY_BANDS["tfidf"])
    if score >= strong_min:
        level = "strong"
    elif score >= mod_min:
        level = "moderate"
    else:
        level = "weak"
    if top_score > 0 and score >= _RELATIVE_BOOST_FRAC * top_score:
        if level == "weak":
            level = "moderate"
        elif level == "moderate":
            level = "strong"
    return level


def _wagner_fisher(s1, s2):
    """Compute the Levenshtein edit distance between two strings.

    Implements the Wagner-Fisher dynamic programming algorithm.  The DP table
    is compressed to a single row to keep memory usage O(min(|s1|, |s2|)).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Integer minimum edit distance (insertions, deletions, substitutions).
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev,
                                dp[j],
                                dp[j - 1])
            prev = temp

    return dp[n]


def _shared_prefix_len(a, b):
    """Length of the longest common prefix of two strings.

    Used as a final tiebreaker after edit distance and LCS in
    ``_correct_token``: when two candidates have identical edit distance
    and identical LCS to the query, the one that agrees with the query
    from the start wins. Fixes cases like ``ender → under`` (LCS ties with
    ``endur`` but prefix agreement is 0 vs 3).
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _lcs_length(s1, s2):
    """Longest common subsequence length between two strings.

    Not to be confused with longest common *substring* — a subsequence is
    allowed to skip characters on either side. Implemented as the standard
    O(m*n) DP with the table compressed to a single row for O(min(m, n))
    memory. Used as a tiebreaker when edit distance is equal for multiple
    spell-correction candidates: ``tcep`` and ``tricep`` share the whole
    subsequence ``tcep`` (LCS=4), while ``tcep`` and ``top`` share only
    ``tp`` (LCS=2), so ``tricep`` correctly wins over the arbitrary
    first-seen choice.
    """
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[n]


def _correct_token(token, vocab_by_length, max_dist=2):
    """Return the closest vocabulary term to *token*, or *token* itself.

    To avoid an expensive O(V) scan on every token, candidate vocabulary terms
    are pre-bucketed by length.  Only terms whose length differs from the query
    token by at most *max_dist* characters can possibly be within *max_dist*
    edits, so we restrict the search to those buckets.

    Ranks candidates by ``(edit_distance, -(2*lcs + prefix))``: lower
    edit distance wins, ties broken by a weighted combination where LCS
    stays dominant but shared prefix is strong enough to overrule it on
    close calls. Examples the weights balance:
      - ``ender → endur`` (lcs 4, prefix 3 → 11) over
        ``gender`` (lcs 5, prefix 0 → 10) — prefix wins the close LCS gap.
      - ``biecp → bicep`` (lcs 4, prefix 2 → 10) over
        ``bie`` (lcs 3, prefix 3 → 9) — LCS wins when prefix favours a
        much shorter/wrong candidate.
      - ``tcep → tricep`` (lcs 4 → 9) over ``top`` (lcs 2 → 5) — plain LCS.

    Args:
        token: A single lowercase query token.
        vocab_by_length: Dict mapping token_length → list of vocab terms of
            that length.  Built once in ExerciseSearcher.__init__.
        max_dist: Maximum edit distance to accept a correction (default 2).
            Tokens needing more than this many edits are returned unchanged.

    Returns:
        The best-matching vocabulary term if one is found within *max_dist*
        edits, otherwise the original token.
    """
    if len(token) <= 2:
        return token

    best_term = token
    best_key = (max_dist + 1, 0)  # (edit_dist, -(2*lcs + prefix)); lower is better

    for length in range(len(token) - max_dist, len(token) + max_dist + 1):
        for candidate in vocab_by_length.get(length, []):
            d = _wagner_fisher(token, candidate)
            if d > max_dist or d > best_key[0]:
                continue
            key = (d, -(2 * _lcs_length(token, candidate)
                        + _shared_prefix_len(token, candidate)))
            if key < best_key:
                best_key = key
                best_term = candidate
                if d == 0:
                    return best_term

    return best_term if best_key[0] <= max_dist else token


def _expand_query(query):
    """Expand a user query with muscle group names and equipment synonyms.

    This bridges the vocabulary gap between how users describe their goals
    (e.g. "vertical jump", "bodyweight") and the terminology used in the
    exercise dataset (e.g. "quadriceps", "body only").

    Expansion steps:
        1. Lowercase the raw query.
        2. Check for goal keywords in GOAL_TO_MUSCLES (longest keys first to
           avoid partial matches) and append the mapped muscle names.
        3. Check for equipment keywords in EQUIPMENT_KEYWORDS and append the
           mapped dataset value (skipping None, which means "no filter").

    Args:
        query: The raw natural-language query string from the user.

    Returns:
        A single expanded query string with the original terms plus any
        appended muscle/equipment tokens, ready for TF-IDF vectorization.
    """
    query_lower = query.lower()
    expanded_terms = [query_lower]

    # multi-word goal matching first
    for goal, muscles in sorted(GOAL_TO_MUSCLES.items(), key=lambda x: -len(x[0])):
        if goal in query_lower:
            expanded_terms.extend(muscles)

    # equipment synonyms
    for kw, mapped in EQUIPMENT_KEYWORDS.items():
        if kw in query_lower and mapped is not None:
            expanded_terms.append(mapped)

    return " ".join(expanded_terms)


class ExerciseSearcher:
    """Loads the exercise dataset and builds a TF-IDF index for retrieval.

    On instantiation the class reads every exercise from the Free Exercise DB
    JSON file, constructs a weighted document string for each exercise (see
    ``_build_weighted_doc``), and fits a scikit-learn ``TfidfVectorizer`` over
    the full corpus.  The resulting sparse TF-IDF matrix is stored in memory
    so that subsequent searches only need to vectorize the query and compute
    cosine similarities — no re-indexing required.

    Attributes:
        exercises (list[dict]): The raw list of exercise dicts loaded from JSON.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.  English
            stop words are removed and the vocabulary is capped at 10 000
            features for performance.
        tfidf_matrix (scipy.sparse.csr_matrix): The sparse term-document matrix
            of shape (n_exercises, n_features).
    """

    def __init__(self):
        """Load exercises from disk and build the TF-IDF index.

        Reads ``exercises_free_db.json``, builds a weighted document for every
        exercise, and fits/transforms the TF-IDF vectorizer in a single pass.
        """
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            self.exercises = json.load(f)

        # Canonical muscle vocabulary, harvested from the dataset itself so
        # query→muscle mapping stays in sync with whatever the corpus uses.
        muscle_set = set()
        for ex in self.exercises:
            for m in ex.get("primaryMuscles") or []:
                muscle_set.add(m.lower())
            for m in ex.get("secondaryMuscles") or []:
                muscle_set.add(m.lower())
        self.all_muscles = muscle_set

        docs = [_build_weighted_doc(ex) for ex in self.exercises]
        # stop_words=None because _tokenize_and_stem already filters them;
        # passing stop_words='english' with a custom tokenizer triggers a
        # sklearn UserWarning about inconsistency (stop words aren't stemmed).
        self.vectorizer = TfidfVectorizer(tokenizer=_tokenize_and_stem,
                                          stop_words=None,
                                          max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

        # LSA layer: project exercises into a 100-dim latent space so
        # semantically related terms cluster. Row-normalized once so
        # cosine similarity at query time is a single dot product.
        self.svd, self.svd_matrix_normed = _fit_svd(self.tfidf_matrix)

        # Stem → most-frequent-surface-form map, used to un-stem tags
        # for display (e.g. `tricep` → `triceps`, `explos` → `explosive`).
        self.stem_to_surface = _build_stem_surface_map(docs)

        # NMF topic model — label-only, does not affect retrieval.
        # Pass all known muscle groups as required labels so IDF-suppressed
        # terms like "back" are guaranteed their own topic.
        (self.topic_nmf, self.topic_matrix,
         self.topic_labels, self.topic_baseline) = _fit_topic_nmf(
            self.tfidf_matrix, self.vectorizer, self.stem_to_surface,
            required_labels=sorted(self.all_muscles))

        # Build a length-bucketed vocabulary index for O(1) candidate lookup
        # during spell correction.  Keys are token lengths; values are lists of
        # all vocabulary terms of that length.
        self.vocab_by_length = {}
        for term in self.vectorizer.vocabulary_:
            self.vocab_by_length.setdefault(len(term), []).append(term)

    def _query_muscles(self, query):
        """Muscles the query is *about*, for the muscle-graph visualization.

        Combines two signals already used by the ranker:
          - GOAL_TO_MUSCLES expansion (e.g. "vertical jump" → quads/glutes/...)
          - direct mentions of any muscle name in the corpus vocabulary
        Returned in insertion order, deduplicated, lowercased.
        """
        q = (query or "").lower()
        out = []
        seen = set()
        for goal, muscles in sorted(GOAL_TO_MUSCLES.items(), key=lambda x: -len(x[0])):
            if goal in q:
                for m in muscles:
                    ml = m.lower()
                    if ml not in seen:
                        seen.add(ml)
                        out.append(ml)
        for m in self.all_muscles:
            if m in q and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def search(self, query, k=5, equipment=None, max_level=None, injured_muscles=None, method="tfidf"):
        """Rank exercises against a natural-language query.

        Steps:
            1. Expand the query via ``_expand_query`` (goal → muscles,
               equipment synonyms).
            2. Transform the expanded query into a TF-IDF vector using the
               already-fitted vectorizer.
            3. Compute cosine similarity between the query vector and every
               exercise vector in the pre-built TF-IDF matrix.
            4. Return the top-k exercises sorted by descending similarity,
               excluding any with a score of 0 (no term overlap at all).

        Args:
            query: Raw natural-language query string (e.g. "improve vertical
                   jump for basketball with gym equipment").
            k: Number of top results to return (default 5).

        Returns:
            A list of up to *k* dicts, each containing:
                - name (str): Exercise name.
                - score (float): Cosine similarity score, rounded to 4 decimals.
                - primaryMuscles (list[str]): Primary muscle groups targeted.
                - secondaryMuscles (list[str]): Secondary muscle groups.
                - level (str): Difficulty level (beginner/intermediate/expert).
                - equipment (str | None): Required equipment.
                - category (str): Exercise category (strength, plyometrics, …).
                - instructions (list[str]): Step-by-step instructions.
        """
        # Spell-correct each query token against the TF-IDF vocabulary.
        # Stem and remove stop words BEFORE spell correction so that valid
        # words like "triceps" aren't needlessly "corrected" to their stemmed
        # form, and stop words like "that" aren't mangled into "th".
        raw_tokens = query.lower().split()
        corrected_display = []  # for "Did you mean?" (human-readable)
        any_corrected = False

        for raw in raw_tokens:
            if raw in _STOP_WORDS or len(raw) <= 1:
                corrected_display.append(raw)
                continue
            aliased = _apply_alias(raw)
            # English-wordlist gate: if the user typed a real English word,
            # don't spell-correct against the corpus vocab. Prevents common
            # queries like "build bigger calves" from being rewritten to
            # "bulk finger calves" just because "build" and "bigger" aren't
            # in the exercise corpus. Stemming is deferred until after this
            # check so real English words skip the Porter pass entirely.
            if aliased in _ENGLISH_WORDS:
                if aliased != raw:
                    corrected_display.append(aliased)
                    any_corrected = True
                else:
                    corrected_display.append(raw)
                continue
            stemmed = _stem(aliased)
            corrected = _correct_token(stemmed, self.vocab_by_length)
            if corrected != stemmed:
                corrected_display.append(corrected)
                any_corrected = True
            elif aliased != raw:
                corrected_display.append(aliased)
                any_corrected = True
            else:
                corrected_display.append(raw)

        corrected_query = " ".join(corrected_display)

        expanded = _expand_query(corrected_query)
        query_vec = self.vectorizer.transform([expanded])
        query_topic_vec = (self.topic_nmf.transform(query_vec)[0]
                           if method == "svd" else None)
        if method == "svd":
            scores = _svd_scores(self.svd, self.svd_matrix_normed, query_vec)
        else:
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply optional filters as a pre-ranking mask. Masked-out rows get a
        # negative score so they never enter the top-k.
        equipment_set = {e.strip().lower() for e in equipment} if equipment else None
        _LEVEL_ORDER = ["beginner", "intermediate", "expert"]
        allowed_levels = None
        if max_level:
            ml = max_level.strip().lower()
            if ml in _LEVEL_ORDER:
                allowed_levels = set(_LEVEL_ORDER[: _LEVEL_ORDER.index(ml) + 1])
        injured_set = {m.strip().lower() for m in injured_muscles} if injured_muscles else None

        if equipment_set or allowed_levels or injured_set:
            for i, ex in enumerate(self.exercises):
                if equipment_set is not None:
                    eq = ex.get("equipment")
                    if eq is None or eq not in equipment_set:
                        scores[i] = -1.0
                        continue
                if allowed_levels is not None:
                    if ex.get("level") not in allowed_levels:
                        scores[i] = -1.0
                        continue
                if injured_set is not None:
                    muscles = set(ex.get("primaryMuscles") or []) | set(ex.get("secondaryMuscles") or [])
                    if not injured_set.isdisjoint(muscles):
                        scores[i] = -1.0
                        continue

        top_indices = np.argsort(scores)[::-1][:k]
        top_score = float(scores.max()) if scores.size else 0.0
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            ex = self.exercises[idx]
            if method == "svd":
                topic_hits = _top_topics(query_topic_vec, self.topic_matrix[idx],
                                         self.topic_baseline)
                tags = [self.topic_labels[t] for t in topic_hits]
                # Guarantee the primary muscle appears — NMF can't always separate
                # co-occurring muscle groups (e.g. middle back vs lats on rows).
                primary = (ex.get("primaryMuscles") or [])
                if primary:
                    m = primary[0].lower()
                    if m not in tags:
                        tags = [m] + tags[:TAGS_PER_RESULT - 1]
            else:
                tags = _top_shared_terms(self.vectorizer, query_vec,
                                         self.tfidf_matrix[idx],
                                         self.stem_to_surface)
            results.append({
                "name": ex.get("name"),
                "score": round(float(scores[idx]), 4),
                "match_quality": _match_quality(float(scores[idx]), top_score, method),
                "tags": tags,
                "primaryMuscles": ex.get("primaryMuscles", []),
                "secondaryMuscles": ex.get("secondaryMuscles", []),
                "level": ex.get("level"),
                "equipment": ex.get("equipment"),
                "category": ex.get("category"),
                "instructions": ex.get("instructions", []),
            })
        return {
            "results": results,
            "corrected_query": corrected_query if any_corrected else None,
            "query_muscles": self._query_muscles(corrected_query or query),
        }


# Module-level singleton — lazily initialized on first call to search() so
# that the JSON loading and TF-IDF fitting only happen once per process
# lifetime, not on every request.
_searcher = None


def search(query, k=5, equipment=None, max_level=None, injured_muscles=None, method="tfidf"):
    """Public entry point: search exercises by natural-language query.

    Lazily initializes a singleton ``ExerciseSearcher`` on first call, then
    delegates to its ``search`` method.  This is the function imported by
    ``routes.py`` and exposed via the ``POST /api/search`` endpoint.

    Args:
        query: Raw query string from the user.
        k: Number of top results to return (default 5).

    Returns:
        A dict with keys:
            - "results": list of result dicts (see ExerciseSearcher.search).
            - "corrected_query": corrected query string if any token was
              spell-corrected, otherwise None.
    """
    global _searcher
    if _searcher is None:
        _searcher = ExerciseSearcher()
    return _searcher.search(
        query,
        k=k,
        equipment=equipment,
        max_level=max_level,
        injured_muscles=injured_muscles,
        method=method,
    )


# ── Program retrieval ────────────────────────────────────────────────────────
# Separate TF-IDF index over the 2,598 unique workout programs in
# programs_cleaned.csv. Uses the same tokenizer / stemmer / spell-correction
# primitives as ExerciseSearcher so query preprocessing is identical across
# the two search surfaces.

PROGRAM_FIELD_WEIGHTS = {
    "title": 3,
    "goal": 3,
    "exercises": 2,
    "description": 1,
}


def _parse_json_list(raw):
    """Parse a JSON-encoded list column from programs_cleaned.csv.

    The CSV round-trips ``json.dumps`` lists (e.g. ``["Powerlifting",
    "Bodybuilding"]``), so ``json.loads`` is the right parser. Falls back to
    an empty list on any failure — downstream code treats missing fields as
    empty space-joined strings.
    """
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def _build_program_doc(row):
    """Build a weighted TF-IDF document string for one program row.

    Field weights mirror ``FIELD_WEIGHTS`` semantics: each field's text is
    repeated ``weight`` times so the TF-IDF vectorizer upweights tokens from
    high-priority fields. ``goal`` and ``exercises`` are JSON lists on disk
    and are space-joined before repetition.
    """
    parts = []
    fields = {
        "title": row.get("title", "") or "",
        "goal": " ".join(_parse_json_list(row.get("goal", ""))),
        "exercises": " ".join(_parse_json_list(row.get("exercises", ""))),
        "description": row.get("description", "") or "",
    }
    for field, weight in PROGRAM_FIELD_WEIGHTS.items():
        text = fields[field]
        if text:
            parts.extend([text] * weight)
    return " ".join(parts)


def _to_float(val):
    """Best-effort float coercion; returns None on failure or empty input."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_int(val):
    """Best-effort int coercion via float (handles ``"1.0"`` from pandas)."""
    f = _to_float(val)
    return int(f) if f is not None else None


class ProgramSearcher:
    """TF-IDF index over workout programs from ``programs_cleaned.csv``.

    Builds one document per program from the weighted concatenation of
    title, goal, exercises, and description. The week/day/exercise
    breakdown is read from each row's ``schedule_json`` column (compact
    dict precomputed by ``data/clean_programs.py``) and expanded into a
    ``title -> list[schedule entry]`` dict so search results can carry a
    full schedule for the UI to render. The schedule is **not** part of
    the TF-IDF document — it's purely for display.

    Attributes:
        programs (list[dict]): Program rows from programs_cleaned.csv.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer sharing
            ``_tokenize_and_stem`` with ``ExerciseSearcher``.
        tfidf_matrix (scipy.sparse.csr_matrix): (n_programs, n_features).
        schedule_by_title (dict[str, list[dict]]): Week/day exercise
            breakdown per program title.
        vocab_by_length (dict[int, list[str]]): Length-bucketed vocabulary
            for fast Wagner-Fisher spell correction.
    """

    def __init__(self):
        with open(PROGRAMS_CSV_PATH, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            self.programs = list(reader)

        docs = [_build_program_doc(row) for row in self.programs]
        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize_and_stem,
            stop_words=None,
            max_features=10000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

        # LSA layer over the 2,598 programs — same 100-dim projection
        # as ExerciseSearcher so paraphrased queries can still rank.
        self.svd, self.svd_matrix_normed = _fit_svd(self.tfidf_matrix)

        # Stem → most-frequent-surface-form map, used to un-stem tags
        # for display.
        self.stem_to_surface = _build_stem_surface_map(docs)

        # NMF topic model — label-only, does not affect retrieval.
        # Collect all distinct goal categories to guarantee coverage.
        goal_cats = set()
        for row in self.programs:
            for g in _parse_json_list(row.get('goal', '')):
                goal_cats.add(g.lower().strip())
        (self.topic_nmf, self.topic_matrix,
         self.topic_labels, self.topic_baseline) = _fit_topic_nmf(
            self.tfidf_matrix, self.vectorizer, self.stem_to_surface,
            required_labels=sorted(goal_cats))

        self.vocab_by_length = {}
        for term in self.vectorizer.vocabulary_:
            self.vocab_by_length.setdefault(len(term), []).append(term)

        # Expand the precomputed schedule_json column on each program row
        # into the display-only schedule dict. No second-file load — this
        # data ships in programs_cleaned.csv. Not part of retrieval scoring.
        self.schedule_by_title = {}
        for row in self.programs:
            title = row.get("title")
            if not title:
                continue
            raw = row.get("schedule_json") or ""
            if not raw:
                self.schedule_by_title[title] = []
                continue
            try:
                sched = json.loads(raw)
            except (ValueError, TypeError):
                self.schedule_by_title[title] = []
                continue
            entries = []
            for name, occurrences in sched.items():
                for occ in occurrences:
                    week, day, sets, reps, rep_type = (occ + [None] * 5)[:5]
                    entries.append({
                        "week": int(week) if week is not None else None,
                        "day": int(day) if day is not None else None,
                        "exercise_name": name,
                        "sets": float(sets) if sets is not None else None,
                        "reps": float(reps) if reps is not None else None,
                        "rep_type": rep_type or None,
                    })
            self.schedule_by_title[title] = entries

    def search(self, query, k=5, method="tfidf"):
        """Rank programs against a natural-language query.

        Mirrors ``ExerciseSearcher.search``: spell-correct each query token
        against the program vocabulary, TF-IDF transform, score with either
        raw TF-IDF cosine similarity (``method="tfidf"``) or the SVD
        latent-space projection (``method="svd"``), then return the top-k
        non-zero scores with their schedule payload attached.
        """
        raw_tokens = query.lower().split()
        corrected_display = []
        any_corrected = False

        for raw in raw_tokens:
            if raw in _STOP_WORDS or len(raw) <= 1:
                corrected_display.append(raw)
                continue
            aliased = _apply_alias(raw)
            if aliased in _ENGLISH_WORDS:
                if aliased != raw:
                    corrected_display.append(aliased)
                    any_corrected = True
                else:
                    corrected_display.append(raw)
                continue
            stemmed = _stem(aliased)
            corrected = _correct_token(stemmed, self.vocab_by_length)
            if corrected != stemmed:
                corrected_display.append(corrected)
                any_corrected = True
            elif aliased != raw:
                corrected_display.append(aliased)
                any_corrected = True
            else:
                corrected_display.append(raw)

        corrected_query = " ".join(corrected_display)
        query_vec = self.vectorizer.transform([corrected_query])
        query_topic_vec = (self.topic_nmf.transform(query_vec)[0]
                           if method == "svd" else None)
        if method == "svd":
            scores = _svd_scores(self.svd, self.svd_matrix_normed, query_vec)
        else:
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:k]
        top_score = float(scores.max()) if scores.size else 0.0
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            row = self.programs[idx]
            title = row.get("title", "")
            if method == "svd":
                topic_hits = _top_topics(query_topic_vec, self.topic_matrix[idx],
                                         self.topic_baseline)
                tags = [self.topic_labels[t] for t in topic_hits]
            else:
                tags = _top_shared_terms(self.vectorizer, query_vec,
                                         self.tfidf_matrix[idx],
                                         self.stem_to_surface)
            results.append({
                "title": title,
                "description": row.get("description", "") or "",
                "goal": _parse_json_list(row.get("goal", "")),
                "level": row.get("level_primary") or None,
                "program_length_weeks": _to_float(row.get("program_length_weeks")),
                "score": round(float(scores[idx]), 4),
                "match_quality": _match_quality(float(scores[idx]), top_score, method),
                "tags": tags,
                "schedule": self.schedule_by_title.get(title, []),
            })
        return {
            "results": results,
            "corrected_query": corrected_query if any_corrected else None,
        }


_program_searcher = None


def search_programs(query, k=5, method="tfidf"):
    """Public entry point: search workout programs by natural-language query.

    Lazily initializes a singleton ``ProgramSearcher`` on first call. Exposed
    via ``POST /api/search_programs`` in ``routes.py``. ``method`` selects
    between raw TF-IDF cosine (``"tfidf"``) and LSA/SVD cosine (``"svd"``).
    """
    global _program_searcher
    if _program_searcher is None:
        _program_searcher = ProgramSearcher()
    return _program_searcher.search(query, k=k, method=method)

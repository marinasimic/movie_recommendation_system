import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

import sqlite3
import json
import hashlib
import os
import datetime

FILE_PATH = "imdb_movies.csv"
REQUIRED_COLUMNS = {'genre', 'overview', 'crew', 'names'}
WEIGHTS = {
            'genre': 1.0,
            'overview': 2.0,
            'crew': 1.0,
            'orig_lang': 0.5,
            'year': 0.5,
        }


def read_and_parse_data(file_path):
    # Load dataset
    df = pd.read_csv("imdb_movies.csv")

    if not REQUIRED_COLUMNS.issubset(df.columns):
        st.error(f"Dataset must contain the following columns: {REQUIRED_COLUMNS}")
        return None
    
    # Fill missing values
    df.fillna('', inplace=True)

    # Convert date to year
    df['date_x'] = df['date_x'].astype(str).str.strip()
    df['year'] = pd.to_datetime(df['date_x'], dayfirst=True, errors='coerce', infer_datetime_format=True).dt.year
    df['year'] = df['year'].fillna('').astype(str)

    # Combine metadata into a single string
    df['metadata'] = df['genre'] + ' ' + df['overview'] + ' ' + df['crew'] + ' ' + df['names'] + df['orig_lang'] + df['year']

    return df

@st.cache_resource
def load_sbert_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load SBERT once per session."""
    return SentenceTransformer(model_name)

@st.cache_data
def cached_encode(texts_tuple, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
    """
    Cache embeddings for a tuple of strings.
    texts_tuple must be a tuple/list of strings (hashable).
    Persist embeddings to disk under .cache/ for reuse between runs.
    """
    model = load_sbert_model(model_name)
    # disk cache key
    key_src = model_name + "\n" + "\n".join(texts_tuple)
    h = hashlib.md5(key_src.encode()).hexdigest()
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"emb_{h}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    embs = model.encode(list(texts_tuple), convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
    np.save(cache_path, embs)
    return embs

def build_similarity_matrix(df, sbert_model=None):
    """
    Build a combined similarity matrix using:
      - genre: Jaccard similarity on genre sets
      - overview: SBERT embeddings + cosine
      - crew: SBERT embeddings + cosine
      - orig_lang: binary exact-match similarity
      - year: 1 - normalized absolute difference (scaled to [0,1])

    weights: dict mapping column -> float importance (defaults below).
    sbert_model: optional pre-loaded SentenceTransformer to avoid reloading.
    """
    if df is None:
        return None

    n = len(df)
    sims = []
    total_weight = 0.0

    # GENRE: Jaccard similarity
    if 'genre' in df.columns and WEIGHTS.get('genre', 0) > 0:
        genres = df['genre'].fillna('').astype(str).apply(
            lambda s: [g.strip().lower() for g in s.split(',') if g.strip()]
        )
        mlb = MultiLabelBinarizer()
        gmat = mlb.fit_transform(genres)
        # pairwise_distances returns Jaccard distance; similarity = 1 - distance
        jaccard_dist = pairwise_distances(gmat, metric='jaccard')
        genre_sim = 1.0 - jaccard_dist
        sims.append((genre_sim, WEIGHTS['genre']))
        total_weight += WEIGHTS['genre']

    sbert_needed = any(col in df.columns and WEIGHTS.get(col, 0) > 0 for col in ('overview', 'crew'))
    model = sbert_model or (SentenceTransformer('all-MiniLM-L6-v2') if sbert_needed else None)

    # OVERVIEW: SBERT embeddings + cosine similarity
    if 'overview' in df.columns and WEIGHTS.get('overview', 0) > 0:
        texts = tuple(df['overview'].fillna('').astype(str).tolist())
        emb = cached_encode(texts)
        overview_sim = cosine_similarity(emb)
        sims.append((overview_sim, WEIGHTS['overview']))
        total_weight += WEIGHTS['overview']

    # CREW: SBERT embeddings + cosine similarity
    if 'crew' in df.columns and WEIGHTS.get('crew', 0) > 0:
        crew_texts = tuple(df['crew'].fillna('').astype(str).tolist())
        crew_emb = cached_encode(crew_texts)
        crew_sim = cosine_similarity(crew_emb)
        sims.append((crew_sim, WEIGHTS['crew']))
        total_weight += WEIGHTS['crew']

    # ORIG_LANG: binary exact match
    if 'orig_lang' in df.columns and WEIGHTS.get('orig_lang', 0) > 0:
        langs = df['orig_lang'].fillna('').astype(str).values
        lang_sim = (langs[:, None] == langs[None, :]).astype(float)
        sims.append((lang_sim, WEIGHTS['orig_lang']))
        total_weight += WEIGHTS['orig_lang']

    # YEAR: normalized absolute difference
    if 'year' in df.columns and WEIGHTS.get('year', 0) > 0:
        yrs = pd.to_numeric(df['year'], errors='coerce')
        yrs.fillna(yrs.median(), inplace=True)
        arr = yrs.values.astype(float)
        if arr.size == 0:
            year_sim = np.ones((n, n))
        else:
            max_range = arr.max() - arr.min()
            if max_range == 0:
                year_sim = np.ones((n, n))
            else:
                diff = np.abs(arr[:, None] - arr[None, :]) / max_range
                year_sim = 1.0 - diff
                year_sim = np.clip(year_sim, 0.0, 1.0)
        sims.append((year_sim, WEIGHTS['year']))
        total_weight += WEIGHTS['year']

    # Combine weighted similarities
    if not sims:
        combined = np.eye(n)
    else:
        combined = np.zeros((n, n), dtype=float)
        for mat, w in sims:
            combined += mat * w
        if total_weight > 0:
            combined /= total_weight

    np.fill_diagonal(combined, 1.0)
    return combined

def setup_database():
    # Connect to (or create) the database
    conn = sqlite3.connect('recommendations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (selected_movies TEXT PRIMARY KEY, recommended_movies TEXT)''')
    return c, conn


# add new helpers: build feature matrix, build ANN index, query recommendations
@st.cache_data
def build_feature_matrix(df, weights=WEIGHTS, model_name: str = "all-MiniLM-L6-v2"):
    """
    Build combined feature vectors per movie:
     - overview, crew: SBERT embeddings (cached)
     - genre: MultiLabelBinarizer one-hot
     - orig_lang: one-hot
     - year: normalized scalar
    Rows are L2-normalized.
    """
    n = len(df)
    parts = []

    # OVERVIEW embeddings
    if 'overview' in df.columns and weights.get('overview', 0) > 0:
        texts = tuple(df['overview'].fillna('').astype(str).tolist())
        emb_over = cached_encode(texts, model_name=model_name)
        parts.append(emb_over * float(weights['overview']))

    # CREW embeddings
    if 'crew' in df.columns and weights.get('crew', 0) > 0:
        texts = tuple(df['crew'].fillna('').astype(str).tolist())
        emb_crew = cached_encode(texts, model_name=model_name)
        parts.append(emb_crew * float(weights['crew']))

    # GENRE one-hot
    if 'genre' in df.columns and weights.get('genre', 0) > 0:
        genres = df['genre'].fillna('').astype(str).apply(
            lambda s: [g.strip().lower() for g in s.split(',') if g.strip()]
        )
        mlb = MultiLabelBinarizer()
        gmat = mlb.fit_transform(genres)
        parts.append(gmat.astype(float) * float(weights['genre']))

    # ORIG_LANG one-hot
    if 'orig_lang' in df.columns and weights.get('orig_lang', 0) > 0:
        lang_mat = pd.get_dummies(df['orig_lang'].fillna('')).values.astype(float)
        parts.append(lang_mat * float(weights['orig_lang']))

    # YEAR normalized
    if 'year' in df.columns and weights.get('year', 0) > 0:
        yrs = pd.to_numeric(df['year'], errors='coerce')
        yrs.fillna(yrs.median(), inplace=True)
        yr_arr = yrs.values.astype(float)
        if yr_arr.max() - yr_arr.min() == 0:
            year_feat = np.zeros((n, 1))
        else:
            year_feat = ((yr_arr - yr_arr.min()) / (yr_arr.max() - yr_arr.min())).reshape(-1, 1)
        parts.append(year_feat * float(weights['year']))

    if not parts:
        return np.zeros((n, 1), dtype=float)

    # horizontally stack and normalize
    combined = np.hstack(parts).astype(float)
    combined = normalize(combined, axis=1)
    return combined

@st.cache_resource
def build_ann_index(df, weights=WEIGHTS, model_name: str = "all-MiniLM-L6-v2", n_neighbors: int = 50):
    """
    Build and cache a NearestNeighbors index and the combined feature matrix.
    Uses cosine distance; kneighbors returns nearest (including query itself).
    """
    features = build_feature_matrix(df, weights=weights, model_name=model_name)
    # choose n_neighbors reasonably large (we'll filter out selected movies later)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, max(2, len(df))), metric='cosine', n_jobs=-1)
    nn.fit(features)
    return nn, features

def get_recommendations_nn(nn, features, selected_indices, k=5):
    """
    Query ANN for recommendations given one or more selected indices.
    Returns list of recommended indices (excluding selected).
    """
    if not selected_indices:
        return []

    # aggregate selected vectors (mean) then normalize
    query_vec = np.mean(features[selected_indices], axis=0).reshape(1, -1)
    query_vec = normalize(query_vec)

    distances, indices = nn.kneighbors(query_vec, n_neighbors=nn.n_neighbors)
    indices = indices[0].tolist()
    # filter out selected indices and keep top-k
    recs = [i for i in indices if i not in selected_indices]
    return recs[:k]

def get_recommendations_pool(nn, features, selected_indices, pool_size=50):
    """
    Return a ranked pool of candidate (idx, score) tuples excluding selected_indices.
    pool_size: desired number of candidates to keep in the session pool.
    """
    if not selected_indices:
        return []

    query_vec = np.mean(features[selected_indices], axis=0).reshape(1, -1)
    query_vec = normalize(query_vec)

    max_neigh = min(pool_size + len(selected_indices) + 5, features.shape[0])
    distances, indices = nn.kneighbors(query_vec, n_neighbors=max_neigh)
    distances = distances[0]
    indices = indices[0]

    pool = []
    for idx, dist in zip(indices, distances):
        if int(idx) in selected_indices:
            continue
        sim = 1.0 - float(dist)
        pool.append((int(idx), float(sim)))
        if len(pool) >= pool_size:
            break
    return pool

def compute_file_hash(path: str) -> str:
    """Return MD5 hash of a file (streaming)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def initialize_database_from_csv(conn: sqlite3.Connection, csv_path: str):
    """
    Create movies table and metadata table if missing and populate movies
    from CSV only when the CSV changed (hash check). Returns True if DB was
    created/updated, False if up-to-date.
    """
    cur = conn.cursor()
    # create tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            name TEXT,
            genre TEXT,
            overview TEXT,
            crew TEXT,
            orig_lang TEXT,
            year TEXT
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT
        )""")
    conn.commit()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    csv_hash = compute_file_hash(csv_path)
    cur.execute("SELECT v FROM meta WHERE k = 'csv_hash'")
    row = cur.fetchone()
    if row and row[0] == csv_hash:
        # DB already in sync with CSV
        return False

    # load csv and normalize same way as before
    df = pd.read_csv(csv_path)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {REQUIRED_COLUMNS}")
    df.fillna('', inplace=True)
    df['date_x'] = df['date_x'].astype(str).str.strip()
    df['year'] = pd.to_datetime(df['date_x'], dayfirst=True, format='mixed', errors='coerce').dt.year
    df['year'] = df['year'].fillna('').astype(str)
    df['metadata'] = df['genre'].astype(str) + ' ' + df['overview'].astype(str) + ' ' + df['crew'].astype(str) + ' ' + df['names'].astype(str) + ' ' + df['orig_lang'].astype(str) + ' ' + df['year'].astype(str)

    # populate table using dataframe index as movie_id
    rows = []
    for idx, r in df.iterrows():
        rows.append((
            int(idx),
            r.get('names', ''),
            r.get('genre', ''),
            r.get('overview', ''),
            r.get('crew', ''),
            r.get('orig_lang', ''),
            r.get('year', ''),
        ))

    # replace existing rows
    cur.executemany("""
        INSERT OR REPLACE INTO movies (movie_id, name, genre, overview, crew, orig_lang, year)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('csv_hash', ?)", (csv_hash,))
    conn.commit()
    return True

def load_movies_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load movies table into a DataFrame (ordered by movie_id)."""
    df = pd.read_sql_query("SELECT * FROM movies ORDER BY movie_id", conn)
    # ensure same column names used elsewhere
    if 'name' in df.columns and 'names' not in df.columns:
        df = df.rename(columns={'name': 'names'})
    return df

def discard_movie(discarded_key, idx):
    """Callback: mark idx as discarded in session state."""
    st.session_state.setdefault(discarded_key, set())
    st.session_state[discarded_key].add(int(idx))

def create_feedback_table(conn: sqlite3.Connection):
    """Create feedback table for CBR retain phase."""
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            selected_ids TEXT,
            candidate_id INTEGER,
            score REAL,
            breakdown TEXT,
            action TEXT,
            weights TEXT,
            ts TEXT
        )
    ''')
    conn.commit()

def log_feedback(conn: sqlite3.Connection,
                 selected_ids,
                 candidate_id: int,
                 score: float,
                 breakdown: dict,
                 action: str,
                 weights: dict = None):
    """Insert a feedback row into DB (retain)."""
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO feedback
        (selected_ids, candidate_id, score, breakdown, action, weights, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        json.dumps(list(selected_ids)),
        int(candidate_id),
        float(score),
        json.dumps(breakdown or {}),
        action,
        json.dumps(weights or WEIGHTS),
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()

def similarity_breakdown_set(df: pd.DataFrame, selected_indices, candidate_idx, model_name="all-MiniLM-L6-v2"):
    """
    Compute per-attribute similarities between candidate and selected set (averaged).
    Returns dict with genre, overview, crew, orig_lang, year and weighted_score.
    """
    def parse_genre(s):
        return {g.strip().lower() for g in str(s).split(',') if g.strip()}

    # genres
    cand_gen = parse_genre(df.at[candidate_idx, 'genre'])
    sel_gens = [parse_genre(df.at[i, 'genre']) for i in selected_indices]
    def jacc(a,b):
        if not a and not b:
            return 0.0
        u = a|b
        return len(a & b) / len(u) if u else 0.0
    genre_vals = [jacc(cand_gen, s) for s in sel_gens] if sel_gens else [0.0]
    genre_sim = float(np.mean(genre_vals))

    # overview
    cand_over = str(df.at[candidate_idx, 'overview']).strip()
    sel_overs = [str(df.at[i, 'overview']).strip() for i in selected_indices]
    overview_sim = 0.0
    if cand_over or any(sel_overs):
        texts = tuple([cand_over] + sel_overs)
        emb = cached_encode(texts, model_name=model_name)
        overview_sim = float(cosine_similarity(emb[0:1], emb[1:])[0].mean()) if len(sel_overs) > 0 else 0.0

    # crew
    cand_crew = str(df.at[candidate_idx, 'crew']).strip()
    sel_crews = [str(df.at[i, 'crew']).strip() for i in selected_indices]
    crew_sim = 0.0
    if cand_crew or any(sel_crews):
        texts = tuple([cand_crew] + sel_crews)
        emb = cached_encode(texts, model_name=model_name)
        crew_sim = float(cosine_similarity(emb[0:1], emb[1:])[0].mean()) if len(sel_crews) > 0 else 0.0

    # orig_lang
    cand_lang = str(df.at[candidate_idx, 'orig_lang']).strip()
    sel_langs = [str(df.at[i, 'orig_lang']).strip() for i in selected_indices]
    lang_vals = []
    for l in sel_langs:
        if cand_lang and l:
            lang_vals.append(1.0 if cand_lang == l else 0.0)
        else:
            lang_vals.append(0.0)
    orig_lang_sim = float(np.mean(lang_vals)) if lang_vals else 0.0

    # year
    years = pd.to_numeric(df['year'], errors='coerce')
    rng = (years.max() - years.min()) if not np.isnan(years.max()) and not np.isnan(years.min()) else 0.0
    a_y = pd.to_numeric(df.at[candidate_idx, 'year'], errors='coerce')
    year_vals = []
    for y in [pd.to_numeric(df.at[i, 'year'], errors='coerce') for i in selected_indices]:
        if np.isnan(a_y) or np.isnan(y) or rng == 0:
            year_vals.append(0.0)
        else:
            year_vals.append(float(max(0.0, 1.0 - abs(a_y - y) / rng)))
    year_sim = float(np.mean(year_vals)) if year_vals else 0.0

    attrs = {
        'genre': genre_sim,
        'overview': overview_sim,
        'crew': crew_sim,
        'orig_lang': orig_lang_sim,
        'year': year_sim
    }
    total_w = sum(WEIGHTS.get(k, 0) for k in attrs.keys()) or 1.0
    weighted = sum(attrs[k] * WEIGHTS.get(k, 0) for k in attrs.keys()) / total_w
    attrs['weighted_score'] = float(weighted)
    return attrs

def discard_and_log(discarded_key, pool_key, safe_key, idx):
    """Callback: discard candidate, log feedback as 'discarded' and update session state."""
    st.session_state.setdefault(discarded_key, set())
    st.session_state[discarded_key].add(int(idx))
    # log feedback (retain)
    conn = sqlite3.connect('recommendations.db')
    create_feedback_table(conn)
    df = load_movies_from_db(conn)
    selected_indices = st.session_state.get("current_selected_indices", [])
    # score map stored as str keys sometimes
    score_map = st.session_state.get(f"scores_{safe_key}", {})
    score = float(score_map.get(str(idx), score_map.get(idx, 0.0)))
    breakdown = similarity_breakdown_set(df, selected_indices, int(idx))
    log_feedback(conn, selected_indices, int(idx), score, breakdown, action='discarded')
    conn.close()

def accept_and_log(discarded_key, pool_key, safe_key, idx):
    """Callback: accept candidate, log feedback as 'accepted' and remove from pool."""
    # mark accepted (so won't appear again)
    st.session_state.setdefault(discarded_key, set())
    st.session_state[discarded_key].add(int(idx))
    # log feedback
    conn = sqlite3.connect('recommendations.db')
    create_feedback_table(conn)
    df = load_movies_from_db(conn)
    selected_indices = st.session_state.get("current_selected_indices", [])
    score_map = st.session_state.get(f"scores_{safe_key}", {})
    score = float(score_map.get(str(idx), score_map.get(idx, 0.0)))
    breakdown = similarity_breakdown_set(df, selected_indices, int(idx))
    log_feedback(conn, selected_indices, int(idx), score, breakdown, action='accepted')
    conn.close()

def repair_on_discard(discarded_key, pool_key, safe_key, discarded_idx, df, nn, features, selected_indices,
                      overview_thresh=0.85, crew_thresh=0.90):
    """
    Simple repair strategy executed after a discard:
      - mark the item discarded (so it's never shown again)
      - diversify the remaining pool by removing candidates too similar to the discarded one
        (based on overview / crew SBERT cosine similarity thresholds)
      - optionally record the top contributing attribute for diagnostics (stored in session_state)
    """
    # mark discarded
    st.session_state.setdefault(discarded_key, set())
    st.session_state[discarded_key].add(int(discarded_idx))

    # get current pool (indices)
    old_pool = list(st.session_state.get(pool_key, []))
    pool_idxs = [p[0] for p in old_pool if p[0] != int(discarded_idx)]

    if not pool_idxs:
        st.session_state[pool_key] = []
        return

    # build texts: first is discarded overview/crew, then candidate overviews/crew
    try:
        disc_over = str(df.at[int(discarded_idx), 'overview']).strip()
        cand_overs = [str(df.at[i, 'overview']).strip() for i in pool_idxs]
    except Exception:
        # fallback: nothing to repair
        return

    # compute overview similarities (skip if all empty)
    remove_set = set()
    if any(disc_over) or any(cand_overs):
        texts = tuple([disc_over] + cand_overs)
        try:
            emb = cached_encode(texts)
            sims = cosine_similarity(emb[0:1], emb[1:])[0]
            for idx, sim in zip(pool_idxs, sims):
                if sim >= overview_thresh:
                    remove_set.add(int(idx))
        except Exception:
            # encoding may fail; ignore repair in that case
            pass

    # optionally also consider crew similarity to diversify
    try:
        disc_crew = str(df.at[int(discarded_idx), 'crew']).strip()
        cand_crews = [str(df.at[i, 'crew']).strip() for i in pool_idxs]
        if any(disc_crew) or any(cand_crews):
            texts = tuple([disc_crew] + cand_crews)
            try:
                embc = cached_encode(texts)
                sims_c = cosine_similarity(embc[0:1], embc[1:])[0]
                for idx, simc in zip(pool_idxs, sims_c):
                    if simc >= crew_thresh:
                        remove_set.add(int(idx))
            except Exception:
                pass
    except Exception:
        pass

    # apply removal to the stored pool, preserving order and scores for remaining items
    new_pool = [p for p in old_pool if p[0] not in remove_set and p[0] not in st.session_state.get(discarded_key, set())]
    st.session_state[pool_key] = new_pool

    # record diagnostic: top contributing attribute for the discarded item (for session analysis)
    try:
        # compute breakdown of discarded candidate vs selected set
        from math import inf
        # use similarity_breakdown_set if present
        if 'similarity_breakdown_set' in globals():
            breakdown = similarity_breakdown_set(df, selected_indices, int(discarded_idx))
            # compute contribution = sim * weight
            contribs = {k: breakdown.get(k, 0.0) * WEIGHTS.get(k, 0.0) for k in ('genre','overview','crew','orig_lang','year')}
            top_attr = max(contribs.items(), key=lambda x: x[1])[0] if contribs else None
            st.session_state.setdefault(f"repair_info_{safe_key}", {})
            st.session_state[f"repair_info_{safe_key}"]['last_discarded'] = int(discarded_idx)
            st.session_state[f"repair_info_{safe_key}"]['top_attribute'] = top_attr
            st.session_state[f"repair_info_{safe_key}"]['removed_count'] = len(remove_set)
    except Exception:
        pass
# ...existing code...
if __name__ == "__main__":
    # Title of the app
    st.title("🎬 Movie Recommender")

    db, conn = setup_database()

    # ensure database exists and is populated from CSV (only when CSV changed)
    initialize_database_from_csv(conn, FILE_PATH)
    # load movies from database instead of reading CSV each run
    df = load_movies_from_db(conn)

    # build ANN index (cached)
    nn, features = build_ann_index(df, weights=WEIGHTS)

     # Movie selection
    # build human-friendly labels "Title — Genre (Year)" and map them to dataframe indices
    labels = []
    label_to_index = {}
    seen = {}
    for idx, row in df.iterrows():
        base = f"{row['names']} — {row.get('genre','')} ({row.get('year','')})"
        # make label unique if duplicates exist by appending the dataframe index
        count = seen.get(base, 0)
        label = base if count == 0 else f"{base} [{idx}]"
        seen[base] = count + 1
        labels.append(label)
        label_to_index[label] = int(idx)

    selected_labels = st.multiselect("Select movies you liked", labels)

    if selected_labels:
        # map selected labels back to dataframe indices
        selected_indices = [label_to_index[l] for l in selected_labels]
        key = ','.join(sorted(selected_labels))

        # Build or reuse a session pool of ranked candidates for this selection
        # use stable key (md5) instead of built-in hash() to avoid cross-run instability
        safe_key = hashlib.md5(key.encode()).hexdigest()
        pool_key = f"rec_pool_{safe_key}"
        discarded_key = f"discarded_{safe_key}"

        # initialize pool and discarded set on selection change
        if st.session_state.get("rec_for") != key:
            st.session_state[pool_key] = get_recommendations_pool(nn, features, selected_indices, pool_size=200)
            st.session_state[discarded_key] = set()
            st.session_state["rec_for"] = key
            # store current selected indices and scores (for callbacks)
            st.session_state["current_selected_indices"] = selected_indices
            st.session_state[f"scores_{safe_key}"] = {str(i): float(s) for i, s in st.session_state[pool_key]}

        # ensure discarded set exists
        st.session_state.setdefault(discarded_key, set())

        # current pool filtered by discarded ids
        pool = [p for p in st.session_state.get(pool_key, []) if p[0] not in st.session_state.get(discarded_key, set())]

        # display recommendations with on_click callbacks
        desired_k = 5
        st.subheader("🎥 Recommended Movies")
        if not pool:
            st.write("No recommendations found.")
        else:
            for i in range(min(desired_k, len(pool))):
                idx, score = pool[i]
                title = df.at[idx, 'names']
                genre = df.at[idx, 'genre']
                year = df.at[idx, 'year']

                cols = st.columns([6, 1, 1])
                with cols[0]:
                    st.write(f"**{title}** — {genre} ({year}) — score: {score:.3f}")
                with cols[1]:
                    st.button("Accept", key=f"accept_{safe_key}_{idx}", on_click=accept_and_log, args=(discarded_key, pool_key, safe_key, idx))
                with cols[2]:
                    st.button("Discard", key=f"discard_{safe_key}_{idx}", on_click=discard_and_log, args=(discarded_key, pool_key, safe_key, idx))

    # Close the database connection
    conn.close()


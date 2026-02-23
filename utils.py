

# util.py
# Shared utilities for evaluation.py and inference.py
# - JSON parsing + factor indexing
# - embedding wrapper
# - centroid computation (query-based OR passage-mean)
# - factor<->(low,high) mapping
# - pos/bin helpers
# - nearest-centroid prediction helpers

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None

def split_factor_train_test(
    jsonl_path: str,
    out_train_path: str,
    out_test_path: str,
    test_frac: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    raw = json.loads(Path(jsonl_path).read_text(encoding="utf-8"))

    train_records = []
    test_records = []

    for rec in raw:
        f_low  = [t.strip() for t in (rec.get("factor_characteristic_low")  or []) if t.strip()]
        f_high = [t.strip() for t in (rec.get("factor_characteristic_high") or []) if t.strip()]

        rng.shuffle(f_low)
        n_low_test = int(round(len(f_low) * test_frac))
        low_test  = f_low[:n_low_test]
        low_train = f_low[n_low_test:]

        rng.shuffle(f_high)
        n_high_test = int(round(len(f_high) * test_frac))
        high_test  = f_high[:n_high_test]
        high_train = f_high[n_high_test:]

        base = {
            "factor_name": rec["factor_name"],
            "factor_shortage": rec["factor_shortage"],
            "factor_description": rec["factor_description"],
            "provider": rec.get("provider", None),
        }

        if low_train or high_train:
            train_records.append({
                **base,
                "factor_characteristic_low":  low_train,
                "factor_characteristic_high": high_train,
            })

        if low_test or high_test:
            test_records.append({
                **base,
                "factor_characteristic_low":  low_test,
                "factor_characteristic_high": high_test,
            })

    Path(out_train_path).write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(out_test_path).write_text(
        json.dumps(test_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


LOW = "LOW"
HIGH = "HIGH"

Polarity = Literal["LOW", "HIGH"]
CentroidMode = Literal["query", "passage_mean"]
BinMethod = Literal["round", "floor", "ceil"]


# =========================
#   IO / PARSING
# =========================

def load_raw(json_path: str | Path) -> List[Dict[str, Any]]:
    """Loads the factor JSON file (it's actually JSON, despite .jsonl naming in your code)."""
    p = Path(json_path)
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_text_list(xs: Optional[Iterable[str]]) -> List[str]:
    """Strip + drop empty."""
    if not xs:
        return []
    out: List[str] = []
    for t in xs:
        if not t:
            continue
        s = t.strip()
        if s:
            out.append(s)
    return out


def build_factor_index(
    raw: Sequence[Dict[str, Any]],
    factors: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[Polarity, List[str]]]:
    """
    Builds an append-based index:
      index[factor][LOW]  -> list[str]
      index[factor][HIGH] -> list[str]

    Important: if there are multiple records per factor, we append (never overwrite).
    """
    factor_set = set(factors) if factors is not None else None
    index: Dict[str, Dict[Polarity, List[str]]] = {}

    for rec in raw:
        f = rec.get("factor_shortage")
        if not f:
            continue
        if factor_set is not None and f not in factor_set:
            continue

        lows = normalize_text_list(rec.get("factor_characteristic_low"))
        highs = normalize_text_list(rec.get("factor_characteristic_high"))
        if not lows and not highs:
            continue

        slot = index.setdefault(f, {LOW: [], HIGH: []})
        slot[LOW].extend(lows)
        slot[HIGH].extend(highs)

    if not index:
        raise ValueError("No factor passages found in the input file.")
    return index


def list_factors(raw_or_index: Sequence[Dict[str, Any]] | Dict[str, Any]) -> List[str]:
    """Returns sorted unique factors from raw (list of dict) or from index dict."""
    if isinstance(raw_or_index, dict):
        return sorted(raw_or_index.keys())
    return sorted({rec["factor_shortage"] for rec in raw_or_index if rec.get("factor_shortage")})


def collect_passages_with_labels(
    index: Dict[str, Dict[Polarity, List[str]]],
    add_prefix: bool = True,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Flattens factor index into:
      - passages: ["passage: ...", ...] (or raw text if add_prefix=False)
      - labels_factor: ["REF", "REF", ...]
      - labels_factor_arr: np.array
      - labels_polarity_arr: np.array(["LOW"/"HIGH"])
    """
    passages: List[str] = []
    labels_factor: List[str] = []
    labels_pol: List[str] = []

    for f in sorted(index.keys()):
        for pol in (LOW, HIGH):
            for t in index[f][pol]:
                passages.append(f"passage: {t}" if add_prefix else t)
                labels_factor.append(f)
                labels_pol.append(pol)

    if not passages:
        raise ValueError("No passages after collection.")

    return passages, labels_factor, np.array(labels_factor), np.array(labels_pol)


# =========================
#   EMBEDDING
# =========================

def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 64,
    normalize: bool = True,
    progress: bool = False,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Unified wrapper around SentenceTransformer.encode.
    Returns np.ndarray of shape (N, D) with requested dtype.
    """
    if not texts:
        return np.zeros((0, 0), dtype=dtype)

    show_bar = bool(progress and (tqdm is not None))
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=show_bar,
    )
    return np.asarray(emb, dtype=dtype)


# =========================
#   PREFIX HELPERS
# =========================

def to_passage_texts(texts: Sequence[str]) -> List[str]:
    return [f"passage: {t}" for t in texts]


def to_query_texts(labels: Sequence[str]) -> List[str]:
    return [f"query: {lab}" for lab in labels]


def factor_lowhigh_labels(factors: Sequence[str]) -> List[str]:
    """['REF_LOW','REF_HIGH', ...] in (LOW,HIGH) order per factor."""
    return [f"{f}_{pol}" for f in factors for pol in (LOW, HIGH)]


# =========================
#   CENTROIDS
# =========================

def centroids_query_lowhigh(
    model: SentenceTransformer,
    factors: Sequence[str],
    normalize: bool = True,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Returns label->vec where label in {'F_LOW','F_HIGH'} using query embeddings:
      vec = embed('query: F_LOW')
    """
    labs = factor_lowhigh_labels(factors)
    q = to_query_texts(labs)
    emb = encode_texts(model, q, batch_size=batch_size, normalize=normalize, progress=False)
    return {lab: emb[i] for i, lab in enumerate(labs)}


def centroids_passage_mean_lowhigh(
    model: SentenceTransformer,
    index: Dict[str, Dict[Polarity, List[str]]],
    normalize_embeddings: bool = True,
    batch_size: int = 64,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Returns label->vec where label in {'F_LOW','F_HIGH'} by averaging passage embeddings:
      centroid(F_LOW) = mean(embed('passage: ...') for low passages), then normalized.
    """
    texts: List[str] = []
    labels: List[str] = []

    for f in sorted(index.keys()):
        for pol in (LOW, HIGH):
            lab = f"{f}_{pol}"
            for t in index[f][pol]:
                texts.append(f"passage: {t}")
                labels.append(lab)

    if not texts:
        raise ValueError("No texts available to compute passage-mean centroids.")

    embs = encode_texts(
        model,
        texts,
        batch_size=batch_size,
        normalize=normalize_embeddings,
        progress=progress,
    )  # (N, D)

    # Sum per label then normalize
    sum_by: Dict[str, np.ndarray] = {}
    cnt_by: Dict[str, int] = {}

    for lab, e in zip(labels, embs):
        if lab not in sum_by:
            sum_by[lab] = e.astype("float32", copy=True)
            cnt_by[lab] = 1
        else:
            sum_by[lab] += e
            cnt_by[lab] += 1

    centroids: Dict[str, np.ndarray] = {}
    for lab, s in sum_by.items():
        c = s / max(1, cnt_by[lab])
        n = float(np.linalg.norm(c))
        if n > 0:
            c = c / n
        centroids[lab] = c.astype("float32")

    return centroids


def compute_centroids(
    model: SentenceTransformer,
    raw_or_index: Sequence[Dict[str, Any]] | Dict[str, Dict[Polarity, List[str]]],
    mode: CentroidMode = "query",
    factors: Optional[Sequence[str]] = None,
    normalize: bool = True,
    batch_size: int = 64,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Unified centroid factory.

    mode='query':
      - needs factors (or can infer from raw/index)
      - returns query centroids for F_LOW/F_HIGH

    mode='passage_mean':
      - needs index (or raw to build it)
      - returns passage-mean centroids for F_LOW/F_HIGH
    """
    if isinstance(raw_or_index, dict):
        index = raw_or_index
        inferred_factors = sorted(index.keys())
    else:
        raw = raw_or_index
        index = build_factor_index(raw, factors=factors)
        inferred_factors = sorted(index.keys())

    use_factors = list(factors) if factors is not None else inferred_factors
    if not use_factors:
        raise ValueError("No factors available for centroid computation.")

    if mode == "query":
        return centroids_query_lowhigh(
            model=model,
            factors=use_factors,
            normalize=normalize,
            batch_size=batch_size,
        )

    if mode == "passage_mean":
        # passage_mean uses the (possibly filtered) index
        if factors is not None:
            # filter index to factors (append-safe)
            idx2 = {f: index[f] for f in use_factors if f in index}
        else:
            idx2 = index
        return centroids_passage_mean_lowhigh(
            model=model,
            index=idx2,
            normalize_embeddings=True,  # keep cosine-friendly
            batch_size=batch_size,
            progress=progress,
        )

    raise ValueError(f"Unknown centroid mode: {mode}")


def build_factor_to_low_high(
    centroids: Dict[str, np.ndarray],
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Converts {'CSAP_LOW':vec, 'CSAP_HIGH':vec, ...} to:
      {'CSAP': (low_vec, high_vec), ...}
    """
    out: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for lab, vec in centroids.items():
        parts = lab.split("_", 1)
        if len(parts) != 2:
            continue
        factor, pol = parts
        low_v, high_v = out.get(factor, (None, None))
        if pol == LOW:
            low_v = vec
        elif pol == HIGH:
            high_v = vec
        out[factor] = (low_v, high_v)
    return out


def factor_centroids_mean_lowhigh(
    centroids: Dict[str, np.ndarray],
    factors: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Builds factor-level centroid matrix (F x D) by averaging LOW/HIGH centroids per factor.
    Useful for mixed/twofactor evals.

    Returns:
      (factor_labels, mat) where mat shape (F, D)
    """
    f2lh = build_factor_to_low_high(centroids)
    factor_labels = sorted(factors) if factors is not None else sorted(f2lh.keys())

    mats: List[np.ndarray] = []
    labels: List[str] = []
    for f in factor_labels:
        low_v, high_v = f2lh.get(f, (None, None))
        if low_v is None and high_v is None:
            continue
        if low_v is None:
            c = high_v
        elif high_v is None:
            c = low_v
        else:
            c = 0.5 * (low_v + high_v)
            n = float(np.linalg.norm(c))
            if n > 0:
                c = c / n
        mats.append(np.asarray(c, dtype="float32"))
        labels.append(f)

    if not mats:
        raise ValueError("No factor centroids could be built (empty).")

    return labels, np.stack(mats, axis=0)


# =========================
#   POS / BIN HELPERS
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def diff_to_pos_default(diff: float) -> float:
    """
    Default mapping used in your code:
      diff = s_high - s_low
      map [-2..2] -> [1..8] then clamp.
    """
    pos_raw = (diff + 2.0) / 4.0
    pos_raw = clamp(pos_raw, 0.0, 1.0)
    return 1.0 + 7.0 * pos_raw


def diff_to_pos(diff: float, calib: Optional[Tuple[float, float]] = None) -> float:
    """
    Converts diff -> pos (1..8).
    If calib=(a,b) provided: pos = a*diff + b, then clamp to [1,8].
    """
    if calib is not None:
        a, b = calib
        pos = a * diff + b
    else:
        pos = diff_to_pos_default(diff)
    return clamp(float(pos), 1.0, 8.0)


def pos_to_bin(pos: float, method: BinMethod = "round") -> int:
    """
    POS 1..8 -> integer bin 1..8.

    method:
      - 'round': 1.5->2, 2.5->2/3 depending on bankers rounding in python; acceptable for your use
      - 'floor': floor(pos)
      - 'ceil' : ceil(pos)
    """
    p = clamp(float(pos), 1.0, 8.0)
    if method == "round":
        b = int(round(p))
    elif method == "floor":
        b = int(np.floor(p))
    elif method == "ceil":
        b = int(np.ceil(p))
    else:
        raise ValueError(f"Unknown bin method: {method}")
    return int(max(1, min(8, b)))


def bins_around_target(target_bin: int, max_bins: int = 8) -> List[int]:
    """
    Priority order around a target bin:
      target=6 -> [6,5,7,4,8,3,2,1]
    """
    target_bin = int(max(1, min(max_bins, target_bin)))
    bins = list(range(1, max_bins + 1))
    bins.sort(key=lambda b: abs(b - target_bin))
    return bins


# =========================
#   NEAREST CENTROID HELPERS
# =========================

def predict_nearest_labels(
    emb_x: np.ndarray,              # (N, D)
    emb_c: np.ndarray,              # (M, D)
    labels_c: Sequence[str],        # len M
) -> List[str]:
    """Nearest centroid by cosine (dot product if vectors normalized)."""
    if emb_x.ndim != 2 or emb_c.ndim != 2:
        raise ValueError("emb_x and emb_c must be 2D arrays.")
    if emb_x.shape[1] != emb_c.shape[1]:
        raise ValueError("Dimension mismatch between emb_x and emb_c.")
    if len(labels_c) != emb_c.shape[0]:
        raise ValueError("labels_c length must match emb_c rows.")

    sims = emb_x @ emb_c.T
    idx = np.argmax(sims, axis=1)
    return [labels_c[int(i)] for i in idx]


def label_to_factor(label: str) -> str:
    """'CSAP_LOW' -> 'CSAP'"""
    return label.split("_", 1)[0]


def labels_to_factors(labels: Sequence[str]) -> List[str]:
    return [label_to_factor(lab) for lab in labels]


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) == 0:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


# =========================
#   FACTOR SCORE (shared core)
# =========================

def estimate_factor_scores_for_text(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    text: str,
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Shared version of factor score estimation.

    For each factor:
      s_low  = cos(emb(text), centroid(F_LOW))
      s_high = cos(emb(text), centroid(F_HIGH))
      rel    = max(s_low, s_high)
      diff   = s_high - s_low
      pos    = diff_to_pos(diff, calib=factor_calib)
      margin = abs(diff)

    If one centroid is missing: fallback to neutral pos=4.5 and margin=0.
    """
    emb = encode_texts(
        model,
        [f"passage: {text}"],
        batch_size=1,
        normalize=True,
        progress=False,
    )[0]

    f2lh = build_factor_to_low_high(centroids)
    out: Dict[str, Dict[str, float]] = {}

    for f, (c_low, c_high) in f2lh.items():
        if c_low is None and c_high is None:
            continue

        s_low = float(emb @ c_low) if c_low is not None else float("nan")
        s_high = float(emb @ c_high) if c_high is not None else float("nan")

        if c_low is None or c_high is None:
            base = c_low if c_low is not None else c_high
            rel = float(emb @ base) if base is not None else float("nan")
            pos = 4.5
            margin = 0.0
        else:
            rel = max(s_low, s_high)
            diff = s_high - s_low
            margin = abs(diff)
            calib = pos_calib.get(f) if pos_calib is not None else None
            pos = diff_to_pos(diff, calib=calib)

        out[f] = {
            "rel": float(rel),
            "pos": float(pos),
            "s_low": float(s_low),
            "s_high": float(s_high),
            "margin": float(margin),
        }

    return out


# =========================
#   OPTIONAL: lightweight dataclass for 1D axis diagnostics
# =========================

@dataclass
class FactorExample:
    text: str
    polarity: str   # LOW / HIGH
    alpha: float    # projection along factor axis



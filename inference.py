# inference.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import utils as U


LOW = U.LOW
HIGH = U.HIGH


# =========================
#   CENTROIDS (explicit mode)
# =========================

def compute_centroids_for_inference(
    model: SentenceTransformer,
    jsonl_path: str,
    factors: Optional[List[str]] = None,
    centroid_mode: U.CentroidMode = "passage_mean",
    batch_size: int = 64,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Inference-side centroid builder with explicit strategy.
    Default mode is 'passage_mean' (stable on your own dataset),
    but you can switch to 'query' for consistency checks.
    """
    raw = U.load_raw(jsonl_path)
    index = U.build_factor_index(raw, factors=factors)

    use_factors = factors if factors is not None else sorted(index.keys())

    centroids = U.compute_centroids(
        model,
        raw_or_index=index,
        mode=centroid_mode,
        factors=use_factors,
        normalize=True,
        batch_size=batch_size,
        progress=progress,
    )
    return centroids


# =========================
#   FACTOR SCORES (shared util)
# =========================

def estimate_factor_scores_for_text(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    text: str,
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Thin wrapper around util. Keeps inference.py public API stable.
    """
    return U.estimate_factor_scores_for_text(model, centroids, text, pos_calib=pos_calib)


def compute_factor_scores_for_texts(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    texts: List[str],
    pos_calib: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Dict[str, float]]]:
    return [estimate_factor_scores_for_text(model, centroids, t, pos_calib=pos_calib) for t in texts]


# =========================
#   POS CALIBRATION
# =========================

def compute_factor_pos_calibration(
    model: SentenceTransformer,
    centroids: Dict[str, np.ndarray],
    jsonl_path: str,
    factors: Optional[List[str]] = None,
    low_target: float = 2.0,
    high_target: float = 7.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Factor-wise linear calibration for diff -> pos:
      LOW examples mean -> low_target
      HIGH examples mean -> high_target
    """
    raw = U.load_raw(jsonl_path)
    index = U.build_factor_index(raw, factors=factors)

    use_factors = factors if factors is not None else sorted(index.keys())

    # collect examples for calibration
    examples: List[Tuple[str, str, str]] = []  # (factor, polarity, text)
    for f in use_factors:
        for t in index[f][LOW]:
            examples.append((f, LOW, t))
        for t in index[f][HIGH]:
            examples.append((f, HIGH, t))

    if not examples:
        raise ValueError("No examples found for calibration.")

    # collect diffs by factor+polarity
    diff_low: Dict[str, List[float]] = {}
    diff_high: Dict[str, List[float]] = {}

    for f, pol, t in examples:
        fs_all = U.estimate_factor_scores_for_text(model, centroids, t, pos_calib=None)
        fs = fs_all.get(f)
        if not fs:
            continue

        s_low = float(fs.get("s_low", np.nan))
        s_high = float(fs.get("s_high", np.nan))
        if np.isnan(s_low) or np.isnan(s_high):
            continue

        diff = s_high - s_low
        if pol == LOW:
            diff_low.setdefault(f, []).append(diff)
        else:
            diff_high.setdefault(f, []).append(diff)

    calib: Dict[str, Tuple[float, float]] = {}
    for f in sorted(set(list(diff_low.keys()) + list(diff_high.keys()))):
        lows = diff_low.get(f, [])
        highs = diff_high.get(f, [])
        if not lows or not highs:
            continue

        mean_low = float(np.mean(lows))
        mean_high = float(np.mean(highs))
        if abs(mean_high - mean_low) < 1e-4:
            continue

        a = (high_target - low_target) / (mean_high - mean_low)
        b = low_target - a * mean_low
        calib[f] = (a, b)

    return calib


# =========================
#   PRE-BINNING (production bins)
# =========================

def prebin_texts_by_factor(
    texts: List[str],
    all_scores: List[Dict[str, Dict[str, float]]],
    rel_threshold: float = 0.3,
    margin_threshold: float = 0.0,
    top_k_factors: int = 3,
    bin_method: U.BinMethod = "round",
) -> Dict[str, Dict[int, List[Tuple[str, Dict[str, float], Dict[str, Dict[str, float]]]]]]:
    """
    Production binning:
      - uses absolute pos -> bin (1..8)
      - gates by rel and margin
      - only keeps (text,factor) if factor in top_k_factors by rel for that text
    """
    by_factor: Dict[str, Dict[int, List[Tuple[str, Dict[str, float], Dict[str, Dict[str, float]]]]]] = {}

    for text, fs_all in zip(texts, all_scores):
        # rank factors by relevance
        ordered = sorted(fs_all.items(), key=lambda kv: -float(kv[1].get("rel", 0.0)))
        top_factors = [f for f, _ in ordered[:top_k_factors]]

        for factor, fs in fs_all.items():
            if factor not in top_factors:
                continue

            rel = float(fs.get("rel", 0.0))
            if rel < rel_threshold:
                continue

            margin = float(fs.get("margin", 0.0))
            if margin < margin_threshold:
                continue

            pos = float(fs.get("pos", 4.5))
            bin_id = U.pos_to_bin(pos, method=bin_method)

            factor_bins = by_factor.setdefault(factor, {})
            bucket = factor_bins.setdefault(bin_id, [])
            bucket.append((text, fs, fs_all))

    # sort each bucket by factor relevance
    for factor, bins in by_factor.items():
        for b, bucket in bins.items():
            bins[b] = sorted(bucket, key=lambda triple: -float(triple[1].get("rel", 0.0)))

    return by_factor


# =========================
#   PROFILE SAMPLING (production)
# =========================

def _profile_alignment_key(
    profile_levels: Dict[str, float],
    fs_all: Dict[str, Dict[str, float]],
    rel_threshold: float = 0.3,
) -> Tuple[int, float, float]:
    """
    Simple profile-alignment signature:
      (n_match, mean_abs_diff, mean_rel)
    """
    diffs: List[float] = []
    rels: List[float] = []

    for factor, level in profile_levels.items():
        fs = fs_all.get(factor)
        if fs is None:
            continue

        rel = float(fs.get("rel", 0.0))
        if rel < rel_threshold:
            continue

        pos = float(fs.get("pos", 4.5))
        diffs.append(abs(pos - level))
        rels.append(rel)

    if not diffs:
        return (0, float("inf"), 0.0)

    return (len(diffs), float(np.mean(diffs)), float(np.mean(rels)))


def _transform_alignment_key(k: Tuple[int, float, float]) -> Tuple[int, float, float]:
    n_match, mean_diff, mean_rel = k
    return (-n_match, mean_diff, -mean_rel)


def sample_texts_for_profile_simple(
    profile_levels: Dict[str, float],
    texts: List[str],
    all_scores: List[Dict[str, Dict[str, float]]],
    rel_threshold: float = 0.3,
    margin_threshold: float = 0.0,
    top_k_factors: int = 3,
    n_extreme: int = 3,
    n_mid: int = 1,
    rerank_by_profile: bool = True,
    rerank_rel_threshold: float = 0.3,
    min_rel_for_rerank: Optional[float] = 0.6,
    top_n_per_bin_before_rerank: Optional[int] = 5,
    bin_method: U.BinMethod = "round",
) -> Dict[str, List[Tuple[str, int, Dict[str, float], str]]]:
    """
    Production sampler:
      - bins by absolute pos (1..8)
      - selects candidate bins around target
      - optional rerank by profile alignment
    """
    by_factor = prebin_texts_by_factor(
        texts=texts,
        all_scores=all_scores,
        rel_threshold=rel_threshold,
        margin_threshold=margin_threshold,
        top_k_factors=top_k_factors,
        bin_method=bin_method,
    )

    result: Dict[str, List[Tuple[str, int, Dict[str, float], str]]] = {}

    for factor, level in profile_levels.items():
        factor_bins = by_factor.get(factor, {})
        if not factor_bins:
            continue

        selected: List[Tuple[str, int, Dict[str, float], str]] = []

        target_bin = U.pos_to_bin(level, method=bin_method)
        bins_order = U.bins_around_target(target_bin)

        if level <= 3.0:
            # prefer low side first
            bins_order = [b for b in bins_order if b <= 4] + [b for b in bins_order if b > 4]
            needed = n_extreme
            label = "LOW"
        elif level >= 6.0:
            # prefer high side first
            bins_order = [b for b in bins_order if b >= 5] + [b for b in bins_order if b < 5]
            needed = n_extreme
            label = "HIGH"
        else:
            # mid range focus
            bins_order = [b for b in bins_order if 3 <= b <= 6]
            needed = n_mid
            label = "MID"

        picked = 0
        seen_texts = set()

        for b in bins_order:
            if picked >= needed:
                break
            bucket = factor_bins.get(b, [])
            if not bucket:
                continue

            candidates = bucket

            # optionally filter by strong factor relevance
            if min_rel_for_rerank is not None:
                filtered = [x for x in candidates if float(x[1].get("rel", 0.0)) >= min_rel_for_rerank]
                if filtered:
                    candidates = filtered

            # optionally cap to top-N by factor relevance before rerank
            if top_n_per_bin_before_rerank is not None and len(candidates) > top_n_per_bin_before_rerank:
                candidates = sorted(candidates, key=lambda x: -float(x[1].get("rel", 0.0)))[:top_n_per_bin_before_rerank]

            # optionally rerank by whole-profile alignment
            if rerank_by_profile:
                candidates = sorted(
                    candidates,
                    key=lambda triple: _transform_alignment_key(
                        _profile_alignment_key(profile_levels, triple[2], rel_threshold=rerank_rel_threshold)
                    ),
                )

            for text, fs_factor, fs_all in candidates:
                if picked >= needed:
                    break
                if text in seen_texts:
                    continue
                selected.append((text, b, fs_factor, label))
                seen_texts.add(text)
                picked += 1

        result[factor] = selected

    return result

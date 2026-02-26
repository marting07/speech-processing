"""HMM-based recognition backends (discrete and continuous)."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


def _left_right_mask(n_states: int) -> np.ndarray:
    mask = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        mask[i, i] = 1.0
        if i + 1 < n_states:
            mask[i, i + 1] = 1.0
    return mask


def _row_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den = np.sum(mat, axis=1, keepdims=True)
    den[den < eps] = 1.0
    return mat / den


def _kmeans_codebook(vectors: np.ndarray, k: int, max_iters: int = 30, seed: int = 13) -> np.ndarray:
    """Learn vector codebook by k-means."""
    n, d = vectors.shape
    k = max(2, min(k, n))
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(max_iters):
        dists = np.sum((vectors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            pts = vectors[labels == j]
            if pts.size > 0:
                new_centers[j] = np.mean(pts, axis=0)
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    return centers


def _quantize_sequence(seq: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    dists = np.sum((seq[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(int)


def _discrete_forward(obs: np.ndarray, pi: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    t_len = len(obs)
    n = len(pi)
    alpha = np.zeros((t_len, n), dtype=float)
    c = np.zeros(t_len, dtype=float)

    alpha[0] = pi * b[:, obs[0]]
    c[0] = np.sum(alpha[0]) + 1e-12
    alpha[0] /= c[0]

    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ a) * b[:, obs[t]]
        c[t] = np.sum(alpha[t]) + 1e-12
        alpha[t] /= c[t]

    # With scaled forward, log P(O|lambda) = sum_t log(c_t).
    log_like = np.sum(np.log(c + 1e-12))
    return alpha, c, float(log_like)


def _discrete_backward(obs: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    t_len = len(obs)
    n = a.shape[0]
    beta = np.zeros((t_len, n), dtype=float)
    beta[-1] = 1.0 / c[-1]
    for t in range(t_len - 2, -1, -1):
        beta[t] = (a * b[:, obs[t + 1]][None, :]) @ beta[t + 1]
        beta[t] /= c[t]
    return beta


def _train_discrete_hmm(
    obs_sequences: List[np.ndarray],
    n_states: int,
    n_symbols: int,
    n_iters: int,
) -> Dict[str, np.ndarray]:
    """Train left-right discrete HMM using Baum-Welch."""
    n_states = max(2, n_states)
    mask = _left_right_mask(n_states)

    pi = np.zeros(n_states, dtype=float)
    pi[0] = 1.0

    a = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        if i == n_states - 1:
            a[i, i] = 1.0
        else:
            a[i, i] = 0.6
            a[i, i + 1] = 0.4

    b = np.full((n_states, n_symbols), 1.0 / n_symbols, dtype=float)

    for _ in range(max(1, n_iters)):
        pi_acc = np.zeros_like(pi)
        a_num = np.zeros_like(a)
        a_den = np.zeros(n_states, dtype=float)
        b_num = np.zeros_like(b)
        b_den = np.zeros(n_states, dtype=float)

        for obs in obs_sequences:
            if len(obs) < 2:
                continue
            alpha, c, _ = _discrete_forward(obs, pi, a, b)
            beta = _discrete_backward(obs, a, b, c)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12

            pi_acc += gamma[0]

            for t in range(len(obs) - 1):
                xi = (
                    alpha[t][:, None]
                    * a
                    * b[:, obs[t + 1]][None, :]
                    * beta[t + 1][None, :]
                )
                xi_sum = np.sum(xi) + 1e-12
                xi /= xi_sum
                a_num += xi
                a_den += gamma[t]

            for t, o_t in enumerate(obs):
                b_num[:, o_t] += gamma[t]
                b_den += gamma[t]

        if np.sum(pi_acc) > 0:
            pi = pi_acc / (np.sum(pi_acc) + 1e-12)
        for i in range(n_states):
            if a_den[i] > 0:
                a[i] = a_num[i] / (a_den[i] + 1e-12)
        a *= mask
        a = _row_normalize(a + 1e-12)

        for i in range(n_states):
            if b_den[i] > 0:
                b[i] = b_num[i] / (b_den[i] + 1e-12)
        b = _row_normalize(b + 1e-12)

    return {"pi": pi, "a": a, "b": b}


def _score_discrete(obs: np.ndarray, model: Dict[str, np.ndarray]) -> float:
    _, _, ll = _discrete_forward(obs, model["pi"], model["a"], model["b"])
    return ll


def _gaussian_diag_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - mu[None, :, :]
    log_det = np.sum(np.log(var + 1e-12), axis=1)
    maha = np.sum((diff ** 2) / (var[None, :, :] + 1e-12), axis=2)
    d = x.shape[1]
    log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det[None, :] + maha)
    return np.exp(log_prob)


def _continuous_forward(x: np.ndarray, pi: np.ndarray, a: np.ndarray, mu: np.ndarray, var: np.ndarray):
    b = _gaussian_diag_pdf(x, mu, var)
    t_len, n = b.shape
    alpha = np.zeros((t_len, n), dtype=float)
    c = np.zeros(t_len, dtype=float)

    alpha[0] = pi * b[0]
    c[0] = np.sum(alpha[0]) + 1e-12
    alpha[0] /= c[0]
    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ a) * b[t]
        c[t] = np.sum(alpha[t]) + 1e-12
        alpha[t] /= c[t]
    log_like = np.sum(np.log(c + 1e-12))
    return alpha, c, b, float(log_like)


def _continuous_backward(b: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    t_len, n = b.shape
    beta = np.zeros((t_len, n), dtype=float)
    beta[-1] = 1.0 / c[-1]
    for t in range(t_len - 2, -1, -1):
        beta[t] = (a * b[t + 1][None, :]) @ beta[t + 1]
        beta[t] /= c[t]
    return beta


def _init_continuous_params(seqs: List[np.ndarray], n_states: int) -> Tuple[np.ndarray, np.ndarray]:
    d = seqs[0].shape[1]
    buckets = [[] for _ in range(n_states)]
    for seq in seqs:
        t_len = seq.shape[0]
        for s in range(n_states):
            start = int(np.floor(s * t_len / n_states))
            end = int(np.floor((s + 1) * t_len / n_states))
            end = max(end, start + 1)
            end = min(end, t_len)
            buckets[s].append(seq[start:end])

    mu = np.zeros((n_states, d), dtype=float)
    var = np.ones((n_states, d), dtype=float) * 1e-2
    for s in range(n_states):
        chunk = np.concatenate(buckets[s], axis=0) if buckets[s] else np.zeros((1, d))
        mu[s] = np.mean(chunk, axis=0)
        var[s] = np.var(chunk, axis=0) + 1e-3
    return mu, var


def _train_continuous_hmm(
    sequences: List[np.ndarray],
    n_states: int,
    n_iters: int,
) -> Dict[str, np.ndarray]:
    n_states = max(2, n_states)
    d = sequences[0].shape[1]
    mask = _left_right_mask(n_states)

    pi = np.zeros(n_states, dtype=float)
    pi[0] = 1.0

    a = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        if i == n_states - 1:
            a[i, i] = 1.0
        else:
            a[i, i] = 0.6
            a[i, i + 1] = 0.4

    mu, var = _init_continuous_params(sequences, n_states)

    for _ in range(max(1, n_iters)):
        pi_acc = np.zeros_like(pi)
        a_num = np.zeros_like(a)
        a_den = np.zeros(n_states, dtype=float)
        mu_num = np.zeros((n_states, d), dtype=float)
        var_num = np.zeros((n_states, d), dtype=float)
        g_den = np.zeros(n_states, dtype=float)

        for x in sequences:
            if x.shape[0] < 2:
                continue
            alpha, c, b, _ = _continuous_forward(x, pi, a, mu, var)
            beta = _continuous_backward(b, a, c)

            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12
            pi_acc += gamma[0]

            for t in range(x.shape[0] - 1):
                xi = (
                    alpha[t][:, None]
                    * a
                    * b[t + 1][None, :]
                    * beta[t + 1][None, :]
                )
                xi /= np.sum(xi) + 1e-12
                a_num += xi
                a_den += gamma[t]

            for s in range(n_states):
                g = gamma[:, s][:, None]
                g_sum = np.sum(g) + 1e-12
                mu_s = np.sum(g * x, axis=0) / g_sum
                mu_num[s] += np.sum(g * x, axis=0)
                var_num[s] += np.sum(g * (x - mu_s) ** 2, axis=0)
                g_den[s] += float(np.sum(g))

        if np.sum(pi_acc) > 0:
            pi = pi_acc / (np.sum(pi_acc) + 1e-12)

        for i in range(n_states):
            if a_den[i] > 0:
                a[i] = a_num[i] / (a_den[i] + 1e-12)
        a *= mask
        a = _row_normalize(a + 1e-12)

        for s in range(n_states):
            if g_den[s] > 0:
                mu[s] = mu_num[s] / (g_den[s] + 1e-12)
                var[s] = var_num[s] / (g_den[s] + 1e-12) + 1e-3

    return {"pi": pi, "a": a, "mu": mu, "var": var}


def _score_continuous(x: np.ndarray, model: Dict[str, np.ndarray]) -> float:
    _, _, _, ll = _continuous_forward(x, model["pi"], model["a"], model["mu"], model["var"])
    return ll


def recognize_with_hmm_discrete(
    sequences_by_label: Dict[str, List[np.ndarray]],
    query: np.ndarray,
    n_states: int = 5,
    codebook_size: int = 32,
    n_iters: int = 10,
) -> Tuple[Optional[str], Optional[float]]:
    """Train per-label discrete HMMs and score query sequence."""
    all_vectors = [seq for seqs in sequences_by_label.values() for seq in seqs if seq.size > 0]
    if not all_vectors or query.size == 0:
        return None, None
    data = np.concatenate(all_vectors, axis=0)
    if data.shape[0] < 2:
        return None, None

    codebook = _kmeans_codebook(data, k=codebook_size)
    best_label = None
    best_score = -np.inf

    q_obs = _quantize_sequence(query, codebook)
    n_symbols = codebook.shape[0]

    for label, seqs in sequences_by_label.items():
        obs_seqs = [_quantize_sequence(s, codebook) for s in seqs if s.shape[0] >= 2]
        if not obs_seqs:
            continue
        s_states = min(max(2, n_states), max(2, min(len(s) for s in obs_seqs) - 1))
        model = _train_discrete_hmm(obs_seqs, n_states=s_states, n_symbols=n_symbols, n_iters=n_iters)
        score = _score_discrete(q_obs, model)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None, None
    return best_label, float(best_score)


def recognize_with_hmm_continuous(
    sequences_by_label: Dict[str, List[np.ndarray]],
    query: np.ndarray,
    n_states: int = 5,
    n_iters: int = 10,
) -> Tuple[Optional[str], Optional[float]]:
    """Train per-label continuous Gaussian HMMs and score query sequence."""
    if query.size == 0:
        return None, None

    best_label = None
    best_score = -np.inf

    for label, seqs in sequences_by_label.items():
        seqs_valid = [s for s in seqs if s.shape[0] >= 2]
        if not seqs_valid:
            continue
        s_states = min(max(2, n_states), max(2, min(s.shape[0] for s in seqs_valid) - 1))
        model = _train_continuous_hmm(seqs_valid, n_states=s_states, n_iters=n_iters)
        score = _score_continuous(query, model)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None, None
    return best_label, float(best_score)

"""Proximity-index backends for isolated speech recognition."""

from __future__ import annotations

import gzip
import pickle
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dictionary_store import extract_entry_feature
from speech_dsp import dtw_distance, dtw_distance_with_path

FEATURE_TYPES = ("mfcc", "bark", "lpc", "wavelet")
INDEX_TYPES = ("bktree", "fqt", "fixed_height_fqt", "permutation", "lsh")


@dataclass
class IndexedWord:
    """Dictionary item used by proximity indexes."""

    item_id: int
    label: str
    feature_type: str
    matrix: np.ndarray
    pooled: np.ndarray
    source_index: int = -1


@dataclass
class QueryMatch:
    """Match result from indexed query."""

    item_id: int
    label: str
    distance: float


def _pool_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    """Create a compact fixed-size embedding for coarse index search."""
    if matrix.size == 0:
        return np.zeros(2, dtype=float)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return np.concatenate([mean, std]).astype(float)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        m = min(a.shape[0], b.shape[0])
        if m == 0:
            return float("inf")
        return float(np.linalg.norm(a[:m] - b[:m]))
    return float(np.linalg.norm(a - b))


class BKTreeIndex:
    """BK-tree using bucketized DTW as metric for incremental insert/query."""

    def __init__(self, bin_size: float = 0.25):
        self.bin_size = max(1e-6, float(bin_size))
        self.root: Optional[Dict[str, Any]] = None

    def _bucket(self, distance: float) -> int:
        return int(round(distance / self.bin_size))

    def insert(self, item: IndexedWord, items: Dict[int, IndexedWord]) -> None:
        if self.root is None:
            self.root = {"item_id": item.item_id, "children": {}}
            return

        node = self.root
        while True:
            node_item = items[node["item_id"]]
            dist = dtw_distance(item.matrix, node_item.matrix)
            bucket = self._bucket(dist)
            child = node["children"].get(bucket)
            if child is None:
                node["children"][bucket] = {"item_id": item.item_id, "children": {}}
                return
            node = child

    def candidates(
        self,
        query_matrix: np.ndarray,
        items: Dict[int, IndexedWord],
        top_k: int = 5,
        candidate_pool: int = 32,
    ) -> List[int]:
        if self.root is None:
            return []

        ranked: List[Tuple[float, int]] = []

        def rec(node: Dict[str, Any], tau: float) -> float:
            node_item = items[node["item_id"]]
            dist = dtw_distance(query_matrix, node_item.matrix)
            ranked.append((dist, node_item.item_id))
            ranked.sort(key=lambda x: x[0])
            if len(ranked) > max(candidate_pool, top_k):
                del ranked[max(candidate_pool, top_k):]
            if len(ranked) >= top_k:
                tau = ranked[min(top_k - 1, len(ranked) - 1)][0]

            center = self._bucket(dist)
            radius = int(np.ceil((tau if np.isfinite(tau) else dist + 1.0) / self.bin_size)) + 1
            for key, child in node["children"].items():
                if not np.isfinite(tau) or abs(key - center) <= radius:
                    tau = rec(child, tau)
            return tau

        rec(self.root, float("inf"))
        return [item_id for _, item_id in ranked[:candidate_pool]]

    def to_state(self) -> Dict[str, Any]:
        return {"bin_size": self.bin_size, "root": self.root}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BKTreeIndex":
        out = cls(bin_size=state.get("bin_size", 0.25))
        out.root = state.get("root")
        return out


class _FQTNode:
    """Recursive FQT node."""

    def __init__(self, pivot_id: int):
        self.pivot_id = pivot_id
        self.edges: List[Tuple[float, float]] = []
        self.children: List["_FQTNode"] = []
        self.leaf_ids: List[int] = []


class FQTIndex:
    """Feature Quantization Tree over pooled vectors."""

    def __init__(self, branching: int = 4, leaf_size: int = 6, fixed_height: Optional[int] = None):
        self.branching = max(2, int(branching))
        self.leaf_size = max(2, int(leaf_size))
        self.fixed_height = fixed_height
        self.root: Optional[_FQTNode] = None

    def build(self, item_ids: Sequence[int], items: Dict[int, IndexedWord]) -> None:
        ids = list(item_ids)
        self.root = self._build_node(ids, items, depth=0)

    def _build_node(self, item_ids: List[int], items: Dict[int, IndexedWord], depth: int) -> Optional[_FQTNode]:
        if not item_ids:
            return None
        pivot_id = item_ids[0]
        node = _FQTNode(pivot_id)
        if len(item_ids) <= self.leaf_size:
            node.leaf_ids = list(item_ids)
            return node
        if self.fixed_height is not None and depth >= self.fixed_height:
            node.leaf_ids = list(item_ids)
            return node

        rest = item_ids[1:]
        pivot = items[pivot_id].pooled
        dists = np.array([_euclidean(items[i].pooled, pivot) for i in rest], dtype=float)
        if dists.size == 0:
            node.leaf_ids = list(item_ids)
            return node

        qs = [q / self.branching for q in range(1, self.branching)]
        boundaries = sorted(float(np.quantile(dists, q)) for q in qs)
        edges = []
        lo = -1e-12
        for b in boundaries:
            hi = b + 1e-12
            edges.append((lo, hi))
            lo = b
        edges.append((lo, float("inf")))

        buckets: List[List[int]] = [[] for _ in edges]
        for item_id, dist in zip(rest, dists):
            for idx, (e_lo, e_hi) in enumerate(edges):
                if e_lo <= dist < e_hi:
                    buckets[idx].append(item_id)
                    break

        node.edges = edges
        for bucket in buckets:
            child = self._build_node(bucket, items, depth + 1)
            node.children.append(child if child is not None else _FQTNode(pivot_id))
        return node

    def insert(self, item: IndexedWord, items: Dict[int, IndexedWord]) -> None:
        # Rebuild for consistency; this keeps implementation simple and deterministic.
        self.build(list(items.keys()), items)

    def candidates(
        self,
        query_matrix: np.ndarray,
        items: Dict[int, IndexedWord],
        top_k: int = 5,
        candidate_pool: int = 32,
    ) -> List[int]:
        if self.root is None:
            return []
        q = _pool_feature_matrix(query_matrix)
        out: List[int] = []

        def rec(node: _FQTNode, depth: int = 0) -> None:
            if node is None:
                return
            if node.pivot_id not in out:
                out.append(node.pivot_id)
            if node.leaf_ids:
                for item_id in node.leaf_ids:
                    if item_id not in out:
                        out.append(item_id)
                return

            pivot = items[node.pivot_id].pooled
            d = _euclidean(q, pivot)
            order: List[Tuple[float, int]] = []
            for idx, (lo, hi) in enumerate(node.edges):
                center = (lo + hi) * 0.5 if np.isfinite(hi) else lo
                order.append((abs(d - center), idx))
            order.sort(key=lambda x: x[0])
            # Explore nearest bins first, then one adjacent bin for recall.
            for _, idx in order[: min(2, len(order))]:
                child = node.children[idx]
                if child is not None:
                    rec(child, depth + 1)

        rec(self.root)
        if len(out) > candidate_pool:
            out = out[:candidate_pool]
        return out

    def to_state(self) -> Dict[str, Any]:
        def encode(node: Optional[_FQTNode]) -> Any:
            if node is None:
                return None
            return {
                "pivot_id": node.pivot_id,
                "edges": node.edges,
                "leaf_ids": node.leaf_ids,
                "children": [encode(c) for c in node.children],
            }

        return {
            "branching": self.branching,
            "leaf_size": self.leaf_size,
            "fixed_height": self.fixed_height,
            "root": encode(self.root),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "FQTIndex":
        out = cls(
            branching=state.get("branching", 4),
            leaf_size=state.get("leaf_size", 6),
            fixed_height=state.get("fixed_height"),
        )

        def decode(raw: Any) -> Optional[_FQTNode]:
            if raw is None:
                return None
            node = _FQTNode(raw.get("pivot_id", -1))
            node.edges = [tuple(x) for x in raw.get("edges", [])]
            node.leaf_ids = list(raw.get("leaf_ids", []))
            node.children = [decode(c) for c in raw.get("children", [])]
            return node

        out.root = decode(state.get("root"))
        return out


class PermutationIndex:
    """Permutation-based index inspired by pivot permutation search."""

    def __init__(self, n_pivots: int = 8):
        self.n_pivots = max(4, int(n_pivots))
        self.pivot_ids: List[int] = []
        self.signatures: Dict[int, np.ndarray] = {}

    def _signature(self, vec: np.ndarray, pivots: List[np.ndarray]) -> np.ndarray:
        dists = np.array([_euclidean(vec, p) for p in pivots], dtype=float)
        order = np.argsort(dists)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        return ranks.astype(int)

    def build(self, item_ids: Sequence[int], items: Dict[int, IndexedWord]) -> None:
        ids = list(item_ids)
        if not ids:
            self.pivot_ids = []
            self.signatures = {}
            return
        self.pivot_ids = ids[: min(self.n_pivots, len(ids))]
        pivots = [items[i].pooled for i in self.pivot_ids]
        self.signatures = {i: self._signature(items[i].pooled, pivots) for i in ids}

    def insert(self, item: IndexedWord, items: Dict[int, IndexedWord]) -> None:
        if not self.pivot_ids:
            self.build(list(items.keys()), items)
            return
        pivots = [items[i].pooled for i in self.pivot_ids if i in items]
        if len(pivots) < 2:
            self.build(list(items.keys()), items)
            return
        self.signatures[item.item_id] = self._signature(item.pooled, pivots)

    def candidates(
        self,
        query_matrix: np.ndarray,
        items: Dict[int, IndexedWord],
        top_k: int = 5,
        candidate_pool: int = 32,
    ) -> List[int]:
        if not self.signatures:
            return []
        q = _pool_feature_matrix(query_matrix)
        pivots = [items[i].pooled for i in self.pivot_ids if i in items]
        if len(pivots) < 2:
            return list(items.keys())[:candidate_pool]
        qsig = self._signature(q, pivots)
        scored = []
        for item_id, sig in self.signatures.items():
            m = min(len(qsig), len(sig))
            if m == 0:
                continue
            score = float(np.sum(np.abs(qsig[:m] - sig[:m])))
            scored.append((score, item_id))
        scored.sort(key=lambda x: x[0])
        return [item_id for _, item_id in scored[:candidate_pool]]

    def to_state(self) -> Dict[str, Any]:
        return {
            "n_pivots": self.n_pivots,
            "pivot_ids": self.pivot_ids,
            "signatures": {k: v.tolist() for k, v in self.signatures.items()},
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PermutationIndex":
        out = cls(n_pivots=state.get("n_pivots", 8))
        out.pivot_ids = list(state.get("pivot_ids", []))
        out.signatures = {int(k): np.array(v, dtype=int) for k, v in state.get("signatures", {}).items()}
        return out


class LSHIndex:
    """Random-hyperplane LSH over pooled vectors."""

    def __init__(self, n_tables: int = 6, n_planes: int = 12, seed: int = 13):
        self.n_tables = max(1, int(n_tables))
        self.n_planes = max(2, int(n_planes))
        self.seed = int(seed)
        self.dim = 0
        self.hyperplanes: List[np.ndarray] = []
        self.tables: List[Dict[str, set]] = []

    def _ensure(self, dim: int) -> None:
        if self.dim == dim and self.hyperplanes and self.tables:
            return
        self.dim = dim
        rng = np.random.default_rng(self.seed)
        self.hyperplanes = [rng.normal(size=(self.n_planes, dim)).astype(float) for _ in range(self.n_tables)]
        self.tables = [dict() for _ in range(self.n_tables)]

    def _hash(self, vec: np.ndarray, table_idx: int) -> str:
        proj = self.hyperplanes[table_idx] @ vec
        bits = (proj >= 0.0).astype(np.uint8)
        return "".join("1" if b else "0" for b in bits.tolist())

    def build(self, item_ids: Sequence[int], items: Dict[int, IndexedWord]) -> None:
        ids = list(item_ids)
        if not ids:
            return
        dim = items[ids[0]].pooled.shape[0]
        self._ensure(dim)
        self.tables = [dict() for _ in range(self.n_tables)]
        for item_id in ids:
            self.insert(items[item_id], items)

    def insert(self, item: IndexedWord, items: Dict[int, IndexedWord]) -> None:
        vec = item.pooled
        self._ensure(vec.shape[0])
        for t in range(self.n_tables):
            code = self._hash(vec, t)
            bucket = self.tables[t].setdefault(code, set())
            bucket.add(item.item_id)

    def candidates(
        self,
        query_matrix: np.ndarray,
        items: Dict[int, IndexedWord],
        top_k: int = 5,
        candidate_pool: int = 32,
    ) -> List[int]:
        if not self.tables:
            return []
        q = _pool_feature_matrix(query_matrix)
        if q.shape[0] != self.dim:
            return list(items.keys())[:candidate_pool]

        cands = set()
        for t in range(self.n_tables):
            code = self._hash(q, t)
            cands.update(self.tables[t].get(code, set()))
        if not cands:
            return list(items.keys())[:candidate_pool]
        out = list(cands)
        if len(out) > candidate_pool:
            out = out[:candidate_pool]
        return out

    def to_state(self) -> Dict[str, Any]:
        return {
            "n_tables": self.n_tables,
            "n_planes": self.n_planes,
            "seed": self.seed,
            "dim": self.dim,
            "hyperplanes": [h.tolist() for h in self.hyperplanes],
            "tables": [{k: sorted(list(v)) for k, v in table.items()} for table in self.tables],
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "LSHIndex":
        out = cls(
            n_tables=state.get("n_tables", 6),
            n_planes=state.get("n_planes", 12),
            seed=state.get("seed", 13),
        )
        out.dim = int(state.get("dim", 0))
        out.hyperplanes = [np.array(h, dtype=float) for h in state.get("hyperplanes", [])]
        out.tables = []
        for table in state.get("tables", []):
            out.tables.append({k: set(v) for k, v in table.items()})
        return out


class ProximityIndexManager:
    """Abstraction layer for reading dictionary words, indexing, persistence and query."""

    def __init__(self, feature_type: str = "mfcc", index_type: str = "bktree"):
        if feature_type not in FEATURE_TYPES:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        if index_type not in INDEX_TYPES:
            raise ValueError(f"Unsupported index type: {index_type}")
        self.feature_type = feature_type
        self.index_type = index_type
        self.items: Dict[int, IndexedWord] = {}
        self.next_id = 1
        self.index = self._make_index(index_type)

    def _make_index(self, index_type: str):
        if index_type == "bktree":
            return BKTreeIndex()
        if index_type == "fqt":
            return FQTIndex(branching=4, leaf_size=6, fixed_height=None)
        if index_type == "fixed_height_fqt":
            return FQTIndex(branching=4, leaf_size=6, fixed_height=3)
        if index_type == "permutation":
            return PermutationIndex(n_pivots=8)
        if index_type == "lsh":
            return LSHIndex(n_tables=6, n_planes=12)
        raise ValueError(f"Unknown index type: {index_type}")

    def reset(self, feature_type: Optional[str] = None, index_type: Optional[str] = None) -> None:
        if feature_type is not None:
            self.feature_type = feature_type
        if index_type is not None:
            self.index_type = index_type
        self.items = {}
        self.next_id = 1
        self.index = self._make_index(self.index_type)

    def _rebuild_if_needed(self) -> None:
        if isinstance(self.index, (FQTIndex, PermutationIndex, LSHIndex)):
            self.index.build(list(self.items.keys()), self.items)

    def add_dictionary_entry(self, entry: Dict[str, Any], source_index: int = -1) -> bool:
        entry_ft = entry.get("feature_type", "mfcc")
        if entry_ft != self.feature_type:
            return False
        label = entry.get("label")
        if not label:
            return False
        mat = extract_entry_feature(entry, self.feature_type)
        if mat is None or mat.ndim != 2 or mat.size == 0:
            return False
        self.add_word(label, mat.astype(float), source_index=source_index)
        return True

    def add_word(self, label: str, matrix: np.ndarray, source_index: int = -1) -> int:
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("Feature matrix must be non-empty 2-D")
        item_id = self.next_id
        self.next_id += 1
        item = IndexedWord(
            item_id=item_id,
            label=str(label),
            feature_type=self.feature_type,
            matrix=matrix.astype(float),
            pooled=_pool_feature_matrix(matrix),
            source_index=int(source_index),
        )
        self.items[item_id] = item
        self.index.insert(item, self.items)
        self._rebuild_if_needed()
        return item_id

    def ingest_dictionary(self, entries: List[Dict[str, Any]]) -> int:
        added = 0
        for idx, entry in enumerate(entries):
            if self.add_dictionary_entry(entry, source_index=idx):
                added += 1
        return added

    def query(self, query_matrix: np.ndarray, top_k: int = 3, candidate_pool: int = 32) -> Dict[str, Any]:
        if query_matrix.ndim != 2 or query_matrix.size == 0:
            raise ValueError("Query matrix must be non-empty 2-D")
        if not self.items:
            return {"matches": [], "candidate_count": 0}

        cands = self.index.candidates(
            query_matrix=query_matrix,
            items=self.items,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )
        if not cands:
            cands = list(self.items.keys())

        scored: List[QueryMatch] = []
        for item_id in cands:
            item = self.items.get(item_id)
            if item is None:
                continue
            try:
                dist = dtw_distance(query_matrix, item.matrix)
            except Exception:
                continue
            scored.append(QueryMatch(item_id=item_id, label=item.label, distance=float(dist)))
        scored.sort(key=lambda m: m.distance)
        top = scored[: max(1, top_k)]

        out: Dict[str, Any] = {
            "matches": [m.__dict__ for m in top],
            "candidate_count": len(cands),
        }

        if self.index_type == "lsh" and top:
            best = self.items[top[0].item_id]
            dist, path = dtw_distance_with_path(query_matrix, best.matrix)
            out["lsh_alignment"] = {
                "best_label": best.label,
                "distance": float(dist),
                "path": path,
                "constellation": build_alignment_constellation(query_matrix, best.matrix, path).tolist(),
            }
        return out

    def get_stats(self) -> Dict[str, Any]:
        """Return lightweight index statistics for UI reporting."""
        label_counts: Dict[str, int] = {}
        total_frames = 0
        feature_dims = 0
        data_bytes = 0
        for item in self.items.values():
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
            if item.matrix.ndim == 2:
                total_frames += int(item.matrix.shape[0])
                feature_dims = max(feature_dims, int(item.matrix.shape[1]))
            data_bytes += int(item.matrix.nbytes + item.pooled.nbytes)
        try:
            index_state_bytes = len(pickle.dumps(self.index.to_state(), protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            index_state_bytes = 0
        return {
            "items": len(self.items),
            "labels": len(label_counts),
            "label_counts": label_counts,
            "feature_dims": feature_dims,
            "total_frames": total_frames,
            "avg_frames_per_item": (float(total_frames) / len(self.items)) if self.items else 0.0,
            "data_bytes": data_bytes,
            "index_state_bytes": int(index_state_bytes),
            "estimated_total_bytes": int(data_bytes + index_state_bytes),
            "feature_type": self.feature_type,
            "index_type": self.index_type,
        }

    def save(self, path: str) -> None:
        state = {
            "version": 1,
            "feature_type": self.feature_type,
            "index_type": self.index_type,
            "next_id": self.next_id,
            "items": {
                item_id: {
                    "item_id": item.item_id,
                    "label": item.label,
                    "feature_type": item.feature_type,
                    "matrix": item.matrix,
                    "pooled": item.pooled,
                    "source_index": item.source_index,
                }
                for item_id, item in self.items.items()
            },
            "index_state": self.index.to_state(),
        }
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=5) as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ProximityIndexManager":
        with gzip.open(path, "rb") as f:
            state = pickle.load(f)
        out = cls(feature_type=state["feature_type"], index_type=state["index_type"])
        out.next_id = int(state.get("next_id", 1))
        out.items = {}
        for item_id, raw in state.get("items", {}).items():
            iid = int(item_id)
            out.items[iid] = IndexedWord(
                item_id=iid,
                label=raw["label"],
                feature_type=raw["feature_type"],
                matrix=np.array(raw["matrix"], dtype=float),
                pooled=np.array(raw["pooled"], dtype=float),
                source_index=int(raw.get("source_index", -1)),
            )
        index_state = state.get("index_state", {})
        if out.index_type == "bktree":
            out.index = BKTreeIndex.from_state(index_state)
        elif out.index_type in ("fqt", "fixed_height_fqt"):
            out.index = FQTIndex.from_state(index_state)
        elif out.index_type == "permutation":
            out.index = PermutationIndex.from_state(index_state)
        elif out.index_type == "lsh":
            out.index = LSHIndex.from_state(index_state)
        else:
            out.index = out._make_index(out.index_type)
            out._rebuild_if_needed()
        return out


def build_alignment_constellation(
    query_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    path: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Create row-wise alignment map for visualization from DTW path."""
    if query_matrix.size == 0 or ref_matrix.size == 0 or not path:
        return np.empty((0, 0), dtype=float)
    dims = min(query_matrix.shape[1], ref_matrix.shape[1])
    width = max(query_matrix.shape[0], ref_matrix.shape[0])
    out = np.zeros((dims, width), dtype=float)

    nq = max(1, query_matrix.shape[0] - 1)
    nr = max(1, ref_matrix.shape[0] - 1)
    for qi, ri in path:
        qi = int(np.clip(qi, 0, query_matrix.shape[0] - 1))
        ri = int(np.clip(ri, 0, ref_matrix.shape[0] - 1))
        col = int(round(0.5 * (qi / nq + ri / nr) * (width - 1)))
        qv = query_matrix[qi, :dims]
        rv = ref_matrix[ri, :dims]
        # Close values produce brighter points.
        score = np.exp(-np.abs(qv - rv))
        out[:, col] = np.maximum(out[:, col], score)
    return out

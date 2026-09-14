"""The geometry module (spec section 11) -- what makes this ours.

Reproduces the Deepam Saha / Dr. Ashani Dasgupta phage-geometry pipeline:

    genome -> 256-D tetranucleotide-frequency vector -> cosine distance ->
    k-nearest-neighbour graph -> connected components swept over k ->
    Ollivier-Ricci curvature -> the negatively curved bridge edges

and exposes it as the three teaching modules the spec names: Fingerprint (7-9),
Islands (9-11), Bridges (11-12).

Two deliberate implementation decisions:

1.  **No GraphRicciCurvature dependency.** The spec flags its trap directly:
    v0.6.1 pins `scipy<=1.13.1`, which collides with a current scientific
    stack, and pulls in `networkit`, a C++ build that is the real installation
    cost. Ollivier-Ricci curvature is an optimal-transport problem on a
    neighbourhood of at most k+1 points, which `scipy.optimize.linprog` solves
    exactly in milliseconds. Implementing it here removes the dependency, the
    version pin and the C++ build, and -- more importantly -- puts the
    definition in front of a Grade 12 student as forty readable lines instead
    of an opaque call.

2.  **There is no "tetranucleotide frequency package".** The spec lists that as
    a classic hallucination site. It is a `Counter` over a sliding window.

The weighted-versus-unweighted caveat from the spec is not a footnote here. It
is computed, returned, and surfaced by default, because it is the single best
demonstration the platform has of why the transcript is the product.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import mannwhitneyu

from .transcript import bound

__all__ = [
    "tetranucleotide_vector", "tnf_matrix", "cosine_distance_matrix",
    "knn_graph", "component_sweep", "ollivier_ricci", "bridge_analysis",
    "KMERS",
]

_BASES = "ACGT"
KMERS: List[str] = ["".join(p) for p in itertools.product(_BASES, repeat=4)]
_KMER_INDEX = {k: i for i, k in enumerate(KMERS)}


# --------------------------------------------------------------------------
# Module 1: Fingerprint (Grades 7-9)
# --------------------------------------------------------------------------

@bound("tetranucleotide_vector", data_args=("sequence",))
def tetranucleotide_vector(sequence: str, name: str = "sequence",
                           canonical: bool = False) -> Dict[str, Any]:
    """The 256-dimensional fingerprint of a genome.

    G5-L11 teaches pattern-matching against a reference set and calls it
    bioinformatics: seven coloured circles, compared position by position. This
    is the same instinct applied to 256 real numbers. There is no package for
    it and there does not need to be -- it is a Counter over a sliding window.

    `canonical` merges each 4-mer with its reverse complement, which is the
    right choice when the sequencing strand is arbitrary. Off by default,
    because the original analysis being reproduced did not use it.
    """
    seq = "".join(c for c in str(sequence).upper() if c in _BASES)
    if len(seq) < 4:
        return {"refused": True, "reason": "Sequence has fewer than 4 usable bases."}

    counts = Counter(seq[i:i + 4] for i in range(len(seq) - 3))
    vec = np.zeros(256, dtype=float)
    for kmer, n in counts.items():
        idx = _KMER_INDEX.get(kmer)
        if idx is not None:
            vec[idx] = n

    if canonical:
        comp = str.maketrans("ACGT", "TGCA")
        merged = np.zeros(256)
        for i, kmer in enumerate(KMERS):
            rc = kmer.translate(comp)[::-1]
            j = _KMER_INDEX[rc]
            merged[min(i, j)] += vec[i]
        vec = merged

    total = vec.sum()
    freq = vec / total if total > 0 else vec

    top = sorted(zip(KMERS, freq), key=lambda kv: -kv[1])[:8]
    gc = float(sum(seq.count(b) for b in "GC") / len(seq))

    return {
        "name": name, "length": len(seq), "gc_content": gc,
        "n_windows": int(total),
        "counts": [int(x) for x in vec],
        "frequencies": [float(x) for x in freq],
        "kmers": KMERS,
        "canonical": bool(canonical),
        "top_kmers": [{"kmer": k, "frequency": float(f)} for k, f in top],
        "expected_uniform": 1 / 256,
        "most_enriched": {"kmer": top[0][0],
                          "fold_over_uniform": float(top[0][1] * 256)} if top else None,
        "teaches": (
            "Every genome has a 4-letter-word habit. If all 256 words were equally likely each "
            "would appear 0.39% of the time; they do not, and how they differ is a fingerprint "
            "specific enough to tell two phage apart without aligning a single base."),
        "plain_language": (
            f"{name} is {len(seq):,} bases long, {gc:.1%} G+C. Its commonest 4-letter word is "
            f"{top[0][0]} at {top[0][1]:.2%} -- {top[0][1] * 256:.1f} times what you would get by "
            f"chance. Those 256 numbers are this genome's fingerprint."),
        "p_value": None,
    }


def tnf_matrix(sequences: Dict[str, str], canonical: bool = False) -> Tuple[List[str], np.ndarray]:
    """Stack fingerprints into an (n_genomes x 256) matrix. Plain helper, no transcript."""
    names, rows = [], []
    for name, seq in sequences.items():
        r = tetranucleotide_vector.__wrapped__(seq, name=name, canonical=canonical)
        if r.get("refused"):
            continue
        names.append(name)
        rows.append(r["frequencies"])
    return names, np.asarray(rows, dtype=float)


@bound("cosine_distance_matrix", data_args=("vectors",))
def cosine_distance_matrix(vectors: Sequence[Sequence[float]],
                           names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Cosine distance between fingerprints -- the matrix the whole analysis rests on.

    The spec's own verification note applies here: the file shipped as
    `Phage_Adjacency_Matrix.csv` is this -- a dense *distance* matrix, not an
    adjacency matrix. This function reproduces it to `atol=1e-9`.
    """
    X = np.asarray(vectors, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"refused": True, "reason": "Need at least two vectors."}
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = X / norms
    sim = np.clip(U @ U.T, -1.0, 1.0)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)

    off = D[~np.eye(D.shape[0], dtype=bool)]
    return {
        "n": int(X.shape[0]), "names": names,
        "matrix": D.tolist(),
        "metric": "cosine distance (1 - cosine similarity)",
        "scale": {"min": float(off.min()), "max": float(off.max()),
                  "median": float(np.median(off)), "mean": float(off.mean())},
        "note": ("These distances are tiny -- order 1e-3 is normal for tetranucleotide vectors, "
                 "because every genome uses all 256 words and they differ only in proportion. "
                 "Their SMALLNESS is exactly why using them as edge weights in a curvature "
                 "calculation gives a different story from using the graph unweighted. See "
                 "`ollivier_ricci`."),
        "p_value": None,
    }


# --------------------------------------------------------------------------
# Module 2: Islands (Grades 9-11) -- the k sweep and the plateau
# --------------------------------------------------------------------------

def _knn_adjacency(D: np.ndarray, k: int, mutual: bool = False) -> sparse.csr_matrix:
    n = D.shape[0]
    k = int(min(max(1, k), n - 1))
    order = np.argsort(D, axis=1)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in order[i, 1:k + 1]:
            rows.append(i); cols.append(int(j)); vals.append(float(D[i, j]))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if mutual:
        mask = A.multiply(A.T > 0)
        return sparse.csr_matrix(mask)
    # Symmetrise by union: i-j is an edge if either lists the other.
    A = A.maximum(A.T)
    return sparse.csr_matrix(A)


@bound("knn_graph", data_args=("distance_matrix",))
def knn_graph(distance_matrix: Sequence[Sequence[float]], k: int = 20,
              names: Optional[List[str]] = None, mutual: bool = False) -> Dict[str, Any]:
    """Build the k-nearest-neighbour graph at one value of k."""
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return {"refused": True, "reason": "Need a square distance matrix with at least 3 points."}
    A = _knn_adjacency(D, k, mutual)
    n_comp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels).tolist()
    coo = sparse.triu(A, k=1).tocoo()
    edges = [{"source": int(i), "target": int(j), "distance": float(w)}
             for i, j, w in zip(coo.row, coo.col, coo.data)]
    return {
        "k": int(k), "n_nodes": int(D.shape[0]), "n_edges": len(edges),
        "mutual": bool(mutual),
        "edges": edges,
        "component_labels": labels.tolist(),
        "n_components": int(n_comp),
        "component_sizes": sorted(sizes, reverse=True),
        "names": names,
        "components_named": (
            [sorted([names[i] for i in range(len(labels)) if labels[i] == c])
             for c in range(n_comp)] if names else None),
        "plain_language": (
            f"At k = {k}, every genome is joined to its {k} closest relatives. That makes "
            f"{len(edges)} connections and leaves {n_comp} separate "
            f"{'island' if n_comp == 1 else 'islands'} of sizes {sorted(sizes, reverse=True)}."),
        "p_value": None,
    }


@bound("component_sweep", data_args=("distance_matrix",))
def component_sweep(distance_matrix: Sequence[Sequence[float]],
                    k_min: int = 3, k_max: int = 40,
                    names: Optional[List[str]] = None,
                    mutual: bool = False) -> Dict[str, Any]:
    """Sweep k and find the stability plateau. This IS the Islands lesson.

        "The lesson is the plateau: a result that survives seventeen parameter
        values is a result; a result that appears at one value is an artefact
        of the setting."

    Returns the component count at every k, the longest run of identical
    partitions, and -- when the plateau splits the data in two -- the named
    membership of each clade, so it can be checked against a reference list.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1] or n < 4:
        return {"refused": True, "reason": "Need a square distance matrix with at least 4 points."}

    k_max = int(min(k_max, n - 1))
    ks = list(range(int(max(1, k_min)), k_max + 1))
    rows, partitions = [], []
    for k in ks:
        A = _knn_adjacency(D, k, mutual)
        nc, labels = connected_components(A, directed=False)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        # Canonical partition signature, independent of label numbering.
        sig = tuple(sorted(tuple(sorted(np.nonzero(labels == c)[0].tolist()))
                           for c in range(nc)))
        rows.append({"k": k, "n_components": int(nc), "sizes": sizes,
                     "n_edges": int(sparse.triu(A, k=1).nnz)})
        partitions.append(sig)

    # Longest run of an identical partition.
    best_len, best_start, best_sig = 0, 0, None
    i = 0
    while i < len(partitions):
        j = i
        while j + 1 < len(partitions) and partitions[j + 1] == partitions[i]:
            j += 1
        if (j - i + 1) > best_len and len(partitions[i]) > 1:
            best_len, best_start, best_sig = j - i + 1, i, partitions[i]
        i = j + 1

    plateau = None
    if best_sig is not None and best_len >= 2:
        members = [list(c) for c in best_sig]
        plateau = {
            "k_from": ks[best_start], "k_to": ks[best_start + best_len - 1],
            "length": best_len,
            "n_components": len(best_sig),
            "sizes": sorted((len(c) for c in best_sig), reverse=True),
            "members": members,
            "members_named": ([sorted(names[i] for i in c) for c in members] if names else None),
        }

    merged_at = next((r["k"] for r in rows if r["n_components"] == 1), None)

    if plateau is None:
        verdict = ("No partition survived more than one value of k. Whatever structure you see at "
                   "any single k is a property of the setting, not of the genomes. That is a real "
                   "and important negative result.")
        sev = "warn"
    elif best_len >= 5:
        verdict = (f"The same split into {plateau['n_components']} groups of sizes "
                   f"{plateau['sizes']} holds unchanged across {best_len} consecutive values of k "
                   f"(k = {plateau['k_from']} to {plateau['k_to']}). A structure that survives "
                   f"{best_len} different settings of the one arbitrary parameter is a property of "
                   f"the data, not of the parameter.")
        sev = "info"
    else:
        verdict = (f"The longest-surviving split lasts only {best_len} values of k "
                   f"({plateau['k_from']}-{plateau['k_to']}). That is weak. Treat it as a "
                   f"hypothesis, not a finding.")
        sev = "warn"

    return {
        "k_values": ks, "rows": rows,
        "component_counts": [r["n_components"] for r in rows],
        "plateau": plateau,
        "merges_at_k": merged_at,
        "n_nodes": int(n), "names": names,
        "severity": sev,
        "teaches": (
            "k is a knob you chose, not something the genomes told you. So the honest question is "
            "never 'what does the graph look like at k = 20?' but 'what survives every k?'"),
        "plain_language": verdict,
        "p_value": None,
    }


# --------------------------------------------------------------------------
# Module 3: Bridges (Grades 11-12) -- Ollivier-Ricci curvature
# --------------------------------------------------------------------------

def _neighbour_measure(A: sparse.csr_matrix, node: int, alpha: float,
                       weighted: bool) -> Tuple[np.ndarray, np.ndarray]:
    """The probability measure m_x: `alpha` stays put, the rest spreads to neighbours."""
    start, end = A.indptr[node], A.indptr[node + 1]
    nbrs = A.indices[start:end]
    if nbrs.size == 0:
        return np.array([node]), np.array([1.0])
    if weighted:
        # Nearer neighbours receive more mass. Inverse-distance, which is the
        # convention GraphRicciCurvature uses for its weighted variant.
        w = 1.0 / np.maximum(A.data[start:end], 1e-12)
    else:
        w = np.ones(nbrs.size)
    w = w / w.sum() * (1.0 - alpha)
    return np.concatenate([[node], nbrs]), np.concatenate([[alpha], w])


def _w1_exact(supp_x, mass_x, supp_y, mass_y, ground: np.ndarray) -> float:
    """Exact 1-Wasserstein distance by linear programming.

    This is the whole of Ollivier-Ricci curvature's machinery. The transport
    problem over two neighbourhoods of at most k+1 points is a small LP that
    HiGHS solves in well under a millisecond, which is why no C++ dependency is
    needed for any graph a school will build.
    """
    nx_, ny_ = len(supp_x), len(supp_y)
    C = ground[np.ix_(supp_x, supp_y)].ravel()
    # Equality constraints: each source ships its mass, each sink receives its mass.
    rows, cols, vals = [], [], []
    for i in range(nx_):
        for j in range(ny_):
            rows.append(i); cols.append(i * ny_ + j); vals.append(1.0)
    for j in range(ny_):
        for i in range(nx_):
            rows.append(nx_ + j); cols.append(i * ny_ + j); vals.append(1.0)
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(nx_ + ny_, nx_ * ny_))
    b_eq = np.concatenate([mass_x, mass_y])
    res = optimize.linprog(C, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return float("nan")
    return float(res.fun)


def _w1_sinkhorn(supp_x, mass_x, supp_y, mass_y, ground: np.ndarray,
                 reg: float = 0.05, iters: int = 300) -> float:
    """Entropic (Sinkhorn) approximation to W1 -- pure numpy, no LP solve.

    Used when a sweep would otherwise need thousands of LPs in a browser tab.
    It is an APPROXIMATION and the result says so: entropic regularisation
    biases W1 upward, which biases curvature downward, so a curvature computed
    this way is a slight underestimate. The platform never silently swaps one
    for the other -- `method` is recorded in the transcript.
    """
    C = ground[np.ix_(supp_x, supp_y)]
    scale = C.max() if C.max() > 0 else 1.0
    K = np.exp(-C / (reg * scale))
    u = np.ones(len(supp_x))
    a, b = np.asarray(mass_x), np.asarray(mass_y)
    for _ in range(iters):
        v = b / np.maximum(K.T @ u, 1e-300)
        u_new = a / np.maximum(K @ v, 1e-300)
        if np.allclose(u_new, u, rtol=1e-9, atol=1e-12):
            u = u_new
            break
        u = u_new
    P = u[:, None] * K * v[None, :]
    return float((P * C).sum())


@bound("ollivier_ricci", data_args=("distance_matrix",))
def ollivier_ricci(distance_matrix: Sequence[Sequence[float]], k: int = 20,
                   alpha: float = 0.5, weighted: bool = False,
                   names: Optional[List[str]] = None,
                   mutual: bool = False, method: str = "exact",
                   denominator: str = "graph") -> Dict[str, Any]:
    """Ollivier-Ricci curvature on every edge of the k-NN graph.

    For an edge (x, y):

        kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y)

    where m_x spreads probability over x's neighbourhood and W1 is the optimal
    transport cost under the graph's shortest-path metric. Negative curvature
    means the two neighbourhoods are hard to move between -- the edge is a
    bottleneck, the only way across. Those are the mosaic recombinants.

    Three knobs decide the answer, and every one of them is recorded:

    * `weighted`  -- are the raw cosine distances the ground metric, or is the
      graph treated as unweighted (every edge one hop)?
    * `denominator` -- is d(x, y) the graph shortest-path distance ("graph"),
      or the raw edge weight ("raw")? With a consistent weighted metric these
      coincide and the whole calculation is scale-invariant. Mixing them is
      NOT scale-invariant, and on distances of order 1e-3 it is what produces
      the huge negative curvatures reported in the literature note.
    * `method`    -- "exact" linear programming, or "sinkhorn" approximation.

    `bridge_analysis` runs several of these combinations side by side, because
    they disagree and the disagreement is a teaching object, not an
    embarrassment.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1] or n < 4:
        return {"refused": True, "reason": "Need a square distance matrix with at least 4 points."}
    if not (0.0 <= alpha < 1.0):
        return {"refused": True, "reason": "alpha must be in [0, 1)."}

    A = _knn_adjacency(D, k, mutual)
    if weighted:
        ground = dijkstra(A, directed=False)
    else:
        unit = A.copy()
        unit.data = np.ones_like(unit.data)
        ground = dijkstra(unit, directed=False, unweighted=True)
    ground = np.nan_to_num(ground, posinf=float(np.nanmax(ground[np.isfinite(ground)])) * 10 + 1.0)

    if method not in ("exact", "sinkhorn"):
        return {"refused": True, "reason": "method must be 'exact' or 'sinkhorn'."}
    if denominator not in ("graph", "raw"):
        return {"refused": True, "reason": "denominator must be 'graph' or 'raw'."}
    solver = _w1_exact if method == "exact" else _w1_sinkhorn

    coo = sparse.triu(A, k=1).tocoo()
    edges = []
    for i, j, w in zip(coo.row, coo.col, coo.data):
        i, j = int(i), int(j)
        sx, mx = _neighbour_measure(A, i, alpha, weighted)
        sy, my = _neighbour_measure(A, j, alpha, weighted)
        d_xy = float(w) if denominator == "raw" else float(ground[i, j])
        if d_xy <= 0:
            continue
        w1 = solver(sx, mx, sy, my, ground)
        kappa = 1.0 - w1 / d_xy if np.isfinite(w1) else float("nan")
        edges.append({"source": i, "target": j,
                      "source_name": names[i] if names else None,
                      "target_name": names[j] if names else None,
                      "distance": float(w), "graph_distance": d_xy,
                      "wasserstein": float(w1), "curvature": float(kappa)})

    kappas = np.array([e["curvature"] for e in edges], dtype=float)
    finite = kappas[np.isfinite(kappas)]

    # Node curvature: the mean over its incident edges.
    node_k: Dict[int, List[float]] = {i: [] for i in range(n)}
    for e in edges:
        if np.isfinite(e["curvature"]):
            node_k[e["source"]].append(e["curvature"])
            node_k[e["target"]].append(e["curvature"])
    node_curv = [float(np.mean(v)) if v else float("nan") for i, v in sorted(node_k.items())]

    return {
        "k": int(k), "alpha": alpha, "weighted": bool(weighted),
        "method": method, "denominator": denominator,
        "n_edges": len(edges), "edges": edges,
        "node_curvature": node_curv,
        "names": names,
        "summary": {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "fraction_negative": float((finite < 0).mean()) if finite.size else float("nan"),
        },
        "solver_note": ("Exact optimal transport by linear programming (HiGHS). No approximation, "
                        "no external curvature library, no scipy version pin."
                        if method == "exact" else
                        "Entropic (Sinkhorn) approximation. Faster, and it biases W1 upward, which "
                        "biases curvature slightly downward. Use 'exact' for anything reported."),
        "weighting_warning": (
            f"Curvature is highly sensitive to how the graph is weighted. This run used the "
            f"{'raw distances' if weighted else 'unweighted graph'} as the ground metric with the "
            f"{'raw edge weight' if denominator == 'raw' else 'graph shortest-path distance'} as "
            f"d(x,y). Change either and the numbers change -- sometimes by orders of magnitude, on "
            f"the same data with the same algorithm. `bridge_analysis` runs the combinations side "
            f"by side so you can see it rather than take our word for it."),
        "plain_language": (
            f"Computed curvature on all {len(edges)} edges at k = {k}"
            f"{' (distance-weighted)' if weighted else ' (unweighted)'}"
            f"{', raw denominator' if denominator == 'raw' else ''}. Mean curvature "
            f"{finite.mean():.3f}, ranging {finite.min():.3f} to {finite.max():.3f}; "
            f"{(finite < 0).mean():.0%} of edges are negatively curved. Negative means the edge is "
            f"a bottleneck -- the only route between two otherwise separate neighbourhoods."
            if finite.size else "No edges could be evaluated."),
        "p_value": None,
    }


@bound("bridge_analysis", data_args=("distance_matrix",))
def bridge_analysis(distance_matrix: Sequence[Sequence[float]],
                    clade_labels: Optional[Sequence[int]] = None,
                    k: Optional[int] = None, alpha: float = 0.5,
                    names: Optional[List[str]] = None,
                    k_min: int = 3, k_max: int = 40,
                    method: str = "exact") -> Dict[str, Any]:
    """The complete Bridges lesson, end to end.

    1. Sweep k and find the plateau (the two clades).
    2. Find the first k at which they connect, and identify those edges.
    3. Compute curvature and test bridge edges against interior edges.
    4. Name the hub genomes that carry most of the bridge.
    5. Run the whole thing weighted AND unweighted, and report both.

    Step 5 is not optional. The spec's honest caveat is the single best
    demonstration the platform has of why the transcript is the product:
    same data, same algorithm, opposite-looking story.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[0] != D.shape[1] or n < 6:
        return {"refused": True, "reason": "Need a square distance matrix with at least 6 points."}

    sweep = component_sweep.__wrapped__(D, k_min=k_min, k_max=min(k_max, n - 1), names=names)
    plateau = sweep.get("plateau")

    # Decide which clade assignment to use: the caller's, or the plateau's.
    if clade_labels is not None:
        labels = np.asarray(clade_labels).astype(int)
        source = "provided by caller"
    elif plateau and plateau["n_components"] == 2:
        labels = np.zeros(n, dtype=int)
        for ci, members in enumerate(plateau["members"]):
            labels[list(members)] = ci
        source = f"the stability plateau (k = {plateau['k_from']}-{plateau['k_to']})"
    else:
        return {"refused": True,
                "reason": ("No two-clade structure was found, and none was supplied. There is "
                           "nothing here for a bridge analysis to bridge -- which is itself the "
                           "answer: this set of genomes does not split in two."),
                "sweep": sweep}

    # The first k at which the two clades touch.
    bridge_k = k
    if bridge_k is None:
        bridge_k = min(k_max, n - 1)
        for kk in range(max(2, k_min), min(k_max, n - 1) + 1):
            A = _knn_adjacency(D, kk)
            coo = sparse.triu(A, k=1).tocoo()
            if any(labels[i] != labels[j] for i, j in zip(coo.row, coo.col)):
                bridge_k = kk
                break

    # Both weightings, always, side by side. Whether they agree is itself the
    # finding, and it is not known in advance: it depends on how uneven the
    # edge weights are in THIS dataset. The caveat below is written from the
    # numbers actually computed, never asserted ahead of them.
    VARIANTS = {
        "unweighted": dict(weighted=False, denominator="graph"),
        "weighted": dict(weighted=True, denominator="graph"),
    }
    results = {}
    for vname, opts in VARIANTS.items():
        weighted = opts["weighted"]
        curv = ollivier_ricci.__wrapped__(D, k=bridge_k, alpha=alpha, names=names,
                                          method=method, **opts)
        edges = curv["edges"]
        cross = [e for e in edges if labels[e["source"]] != labels[e["target"]]
                 and np.isfinite(e["curvature"])]
        within = [e for e in edges if labels[e["source"]] == labels[e["target"]]
                  and np.isfinite(e["curvature"])]
        kc = np.array([e["curvature"] for e in cross])
        kw = np.array([e["curvature"] for e in within])

        test = None
        if kc.size >= 2 and kw.size >= 2:
            mw = mannwhitneyu(kc, kw, alternative="less")
            test = {"test": "Mann-Whitney U (cross < within)",
                    "statistic": float(mw.statistic), "p_value": float(mw.pvalue),
                    "n_cross": int(kc.size), "n_within": int(kw.size)}

        hubs = Counter()
        for e in cross:
            hubs[e["source"]] += 1
            hubs[e["target"]] += 1
        top_hubs = [{"index": int(i), "name": names[i] if names else None, "bridge_edges": int(c)}
                    for i, c in hubs.most_common(5)]

        results[vname] = {
            "weighted": weighted, "denominator": opts["denominator"],
            "n_bridge_edges": len(cross),
            "n_interior_edges": len(within),
            "all_bridges_negative": bool(kc.size > 0 and np.all(kc < 0)),
            "mean_curvature_bridge": float(kc.mean()) if kc.size else float("nan"),
            "mean_curvature_interior": float(kw.mean()) if kw.size else float("nan"),
            "min_curvature": float(np.min(np.concatenate([kc, kw]))) if (kc.size or kw.size) else float("nan"),
            "overall_mean_curvature": curv["summary"]["mean"],
            "test": test,
            "p_value": test["p_value"] if test else None,
            "bridge_edges": cross,
            "hubs": top_hubs,
            "transcript_id": curv.get("transcript_id"),
        }

    u = results["unweighted"]
    w = results["weighted"]

    # How uneven are the edge weights? This is what decides whether weighting
    # matters at all, so compute it rather than guessing.
    A_dbg = _knn_adjacency(D, bridge_k)
    wts = sparse.triu(A_dbg, k=1).tocoo().data
    wts = wts[wts > 0]
    spread_ratio = float(wts.max() / wts.min()) if wts.size else float("nan")

    um, wm = u["overall_mean_curvature"], w["overall_mean_curvature"]
    gap = abs(um - wm)
    if np.isfinite(um) and np.isfinite(wm) and gap > 0.5 * max(abs(um), abs(wm), 1e-12):
        agreement = (f"They DISAGREE: {um:+.3f} against {wm:+.3f}, a gap of {gap:.3f}. "
                     f"Do not report either without saying which one it is.")
    else:
        agreement = (f"On THIS dataset they happen to agree closely ({um:+.3f} against "
                     f"{wm:+.3f}). That is a fact about these particular genomes, not a general "
                     f"guarantee. The edge weights here span a factor of {spread_ratio:.0f}. On a graph with very uneven distances the weighted "
                     f"curvature can run orders of magnitude more negative, because an edge that "
                     f"is much shorter than the routes around it gets divided by a very small "
                     f"number.")

    caveat = (
        f"Curvature depends on how the graph is weighted, so both runs are shown. "
        f"Unweighted (every edge one hop): mean {um:+.3f}, minimum {u['min_curvature']:.3f}. "
        f"Distance-weighted: mean {wm:+.3f}, minimum {w['min_curvature']:.3f}. "
        f"{agreement} "
        f"The edge weights in this graph range from {wts.min():.3g} to {wts.max():.3g}. "
        f"A curvature quoted without its weighting, its k, its alpha and its solver is not a "
        f"result -- which is exactly why this platform hands you the transcript and not just "
        f"the number.")

    primary = u
    hub_txt = ""
    if primary["hubs"]:
        named = ", ".join(
            f"{h['name'] if h['name'] else '#' + str(h['index'])} ({h['bridge_edges']} of them)"
            for h in primary["hubs"][:3])
        hub_txt = (f" The bridge is carried by a handful of genomes: {named}. Those are the ones to "
                   f"look up -- geometry is pointing at which genomes to examine, and biology has "
                   f"to say why.")

    return {
        "clade_source": source,
        "clade_sizes": sorted(np.bincount(labels).tolist(), reverse=True),
        "clade_members": ([sorted(names[i] for i in np.nonzero(labels == c)[0])
                           for c in range(labels.max() + 1)] if names else None),
        "bridge_k": int(bridge_k),
        "sweep": sweep,
        "unweighted": u, "weighted": w,
        "variants": list(results),
        "edge_weight_spread": spread_ratio,
        "weighting_caveat": caveat,
        "p_value": primary["p_value"],
        "teaches": (
            "Curvature did not tell you the biology. It told you WHERE to look. The genomes on the "
            "negatively curved edges are the ones holding two worlds together; going and finding "
            "out why is the actual research, and it is still open."),
        "plain_language": (
            f"Using the clade split from {source}, the two groups first connect at k = {bridge_k} "
            f"through {u['n_bridge_edges']} edges. "
            + (f"Every one of them is negatively curved. " if u["all_bridges_negative"] else
               f"{sum(1 for e in u['bridge_edges'] if e['curvature'] < 0)} of them are negatively "
               f"curved. ")
            + (f"Mean curvature on the bridge is {u['mean_curvature_bridge']:.3f} against "
               f"{u['mean_curvature_interior']:+.3f} inside the clades"
               + (f" (Mann-Whitney, cross < within, p = {u['test']['p_value']:.3g})."
                  if u["test"] else ".")
               if np.isfinite(u["mean_curvature_bridge"]) else "")
            + hub_txt),
        "notes": [caveat],
    }

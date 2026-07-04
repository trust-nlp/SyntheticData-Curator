import json, argparse, math


from typing import List, Dict, Any


import numpy as np


def parse_args():

    ap = argparse.ArgumentParser(
        description="Overlap & rank correlation for IFD vs multiple variants of metricsB"
    )

    ap.add_argument("--input", type=str, required=True, help="原始数据 JSON 路径")

    ap.add_argument(
        "--ratios",
        type=str,
        default="0.05,0.15,0.20",
        help="逗号分隔比例，例如 0.05,0.15,0.20",
    )

    return ap.parse_args()


def parse_ratios(s: str) -> List[float]:

    out = []

    for t in s.split(","):

        t = t.strip()

        if not t:

            continue

        r = float(t)

        if not (0.0 < r <= 1.0):

            raise ValueError(f"ratio must be in (0,1], got {r}")

        out.append(r)

    if not out:

        raise ValueError("empty --ratios")

    return out


def average_rank_with_ties(values: np.ndarray) -> np.ndarray:

    order = np.argsort(-values, kind="mergesort")

    ranks = np.empty_like(order, dtype=np.float64)

    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    vals_sorted = values[order]

    i = 0

    while i < len(values):

        j = i + 1

        while j < len(values) and vals_sorted[j] == vals_sorted[i]:

            j += 1

        if j - i > 1:

            avg_rank = (i + 1 + j) / 2.0

            ranks[order[i:j]] = avg_rank

        i = j

    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:

    rx = average_rank_with_ties(x)

    ry = average_rank_with_ties(y)

    rx_mean, ry_mean = rx.mean(), ry.mean()

    num = np.sum((rx - rx_mean) * (ry - ry_mean))

    den = np.sqrt(np.sum((rx - rx_mean) ** 2) * np.sum((ry - ry_mean) ** 2))

    return float(num / den) if den != 0 else 0.0


def compute_metricB_variants_for_record(
    rec: Dict[str, Any], id2idx: Dict[Any, int], data: List[Dict[str, Any]]
) -> Dict[str, float]:

    fs = rec.get("few_shot_ifd_kmeans5_complexity", []) or []

    sims = rec.get("kmeans5-complexity-sims", []) or []

    nids = rec.get("kmeans5-complexity", []) or []

    m = min(len(fs), len(sims), len(nids))

    if m <= 0:

        return dict(B_base=0.0, B_noW=0.0, B_keepNeg=0.0, B_keepNeg_noW=0.0)

    base_terms, noW_terms, keep_terms, keep_noW_terms = [], [], [], []

    for k in range(m):

        nid = nids[k]

        if nid not in id2idx:

            continue

        nbr = data[id2idx[nid]]

        nbr_ifd = float(nbr.get("ifd", 0.0))

        delta = nbr_ifd - float(fs[k])

        sim = float(sims[k]) if k < len(sims) else 0.0

        w = 1.0 - sim

        if delta > 0:

            base_terms.append(delta * w)

            noW_terms.append(delta)

        keep_terms.append(delta * w)

        keep_noW_terms.append(delta)

    def mean_or_zero(xs: List[float]) -> float:

        return float(np.mean(xs)) if len(xs) > 0 else 0.0

    return dict(
        B_base=mean_or_zero(base_terms),
        B_noW=mean_or_zero(noW_terms),
        B_keepNeg=mean_or_zero(keep_terms),
        B_keepNeg_noW=mean_or_zero(keep_noW_terms),
    )


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:

    if k <= 0:

        return np.array([], dtype=int)

    k = min(k, len(scores))

    idx = np.argpartition(-scores, kth=k - 1)[:k]

    return idx[np.argsort(-scores[idx], kind="mergesort")]


def report_for_variant(
    name: str, metricA: np.ndarray, metricB: np.ndarray, n: int, ratios: List[float]
):

    rho = spearman_rho(metricA, metricB)

    print(f"\n[{name}] Spearman(ifd, {name}) = {rho:.6f}")

    for r in ratios:

        k = max(1, int(round(n * r)))

        A_idx = set(topk_indices(metricA, k).tolist())

        B_idx = set(topk_indices(metricB, k).tolist())

        inter = len(A_idx & B_idx)

        print(f"[{name}] Overlap@{int(r*100)}%: {inter}/{k} = {inter/float(k):.4f}")


def main():

    args = parse_args()

    ratios = parse_ratios(args.ratios)

    with open(args.input, "r", encoding="utf-8") as f:

        data: List[Dict[str, Any]] = json.load(f)

    n = len(data)

    print(f"Loaded {n} records")

    id2idx: Dict[Any, int] = {}

    for i, rec in enumerate(data):

        rid = rec.get("id", i)

        id2idx[rid] = i

    metricA = np.array([float(rec.get("ifd", 0.0)) for rec in data], dtype=np.float64)

    B_base = np.zeros(n, dtype=np.float64)

    B_noW = np.zeros(n, dtype=np.float64)

    B_keepNeg = np.zeros(n, dtype=np.float64)

    B_keepNeg_noW = np.zeros(n, dtype=np.float64)

    for i, rec in enumerate(data):

        res = compute_metricB_variants_for_record(rec, id2idx, data)

        B_base[i] = res["B_base"]

        B_noW[i] = res["B_noW"]

        B_keepNeg[i] = res["B_keepNeg"]

        B_keepNeg_noW[i] = res["B_keepNeg_noW"]

    report_for_variant("B_base", metricA, B_base, n, ratios)

    report_for_variant("B_noW", metricA, B_noW, n, ratios)

    report_for_variant("B_keepNeg", metricA, B_keepNeg, n, ratios)

    report_for_variant("B_keepNeg_noW", metricA, B_keepNeg_noW, n, ratios)


if __name__ == "__main__":

    main()

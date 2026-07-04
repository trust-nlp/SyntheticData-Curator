import argparse


import json


import os


import numpy as np


from tqdm import tqdm


from sklearn.cluster import KMeans


from sentence_transformers import SentenceTransformer


from joblib import Parallel, delayed


import faiss


try:

    import torch

    HAS_TORCH = True


except Exception:

    HAS_TORCH = False


def parse_args():

    ap = argparse.ArgumentParser(
        "Post-process without CrossEncoder: refresh top5 and kmeans(5) on top32. (GPU-ready, Parallel KMeans)"
    )

    ap.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input JSON produced by step1 (must contain quality_score & complexity_score)",
    )

    ap.add_argument("--output_path", type=str, required=True, help="Output JSON path")

    ap.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")

    ap.add_argument(
        "--sbert_batch_size", type=int, default=64, help="Batch size for SBERT encode"
    )

    ap.add_argument(
        "--device",
        type=str,
        default=("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"),
        choices=["cuda", "cpu"],
        help="Device for SBERT encoding",
    )

    ap.add_argument(
        "--faiss_gpu",
        action="store_true",
        help="Use FAISS GPU index (requires faiss-gpu installed)",
    )

    ap.add_argument("--top5", type=int, default=5)

    ap.add_argument(
        "--topN", type=int, default=32, help="candidate pool size used for clustering"
    )

    ap.add_argument("--k", type=int, default=5, help="KMeans clusters")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--save_embed_path",
        type=str,
        default="",
        help="Optional .npy file to save SBERT embeddings for reuse",
    )

    ap.add_argument(
        "--kmeans_jobs",
        type=int,
        default=-1,
        help="joblib n_jobs for per-item KMeans loop",
    )

    ap.add_argument(
        "--kmeans_batch", type=int, default=256, help="joblib batch_size for scheduling"
    )

    ap.add_argument(
        "--kmeans_n_init",
        type=int,
        default=1,
        help="KMeans n_init (reduce from 10 to 1)",
    )

    ap.add_argument(
        "--kmeans_max_iter",
        type=int,
        default=15,
        help="KMeans max_iter (small since C=topN<=32)",
    )

    return ap.parse_args()


def load_data(p):

    with open(p, "r", encoding="utf-8") as f:

        data = json.load(f)

    for i, rec in enumerate(data):

        rec.setdefault("id", i)

    missing = [
        r["id"]
        for r in data
        if ("quality_score" not in r or "complexity_score" not in r)
    ]

    if missing:

        raise ValueError(
            f"Missing quality/complexity scores for ids: {missing[:10]} ... (total {len(missing)})"
        )

    return data


def build_embeddings(data, sbert_model, device="cpu", batch_size=64):

    model = SentenceTransformer(sbert_model, device=device)

    inst = [rec.get("instruction", "") for rec in data]

    emb = model.encode(
        inst,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return emb


def faiss_search(emb, topN, use_gpu=False):

    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)

    if use_gpu:

        res = faiss.StandardGpuResources()

        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(emb.astype(np.float32, copy=False))

    sims, idxs = index.search(emb.astype(np.float32, copy=False), topN + 1)

    return sims, idxs


def _per_item_kmeans(
    i, data, emb, idxs_all_row, sims_all_row, topN, k, seed, km_n_init, km_max_iter
):

    ids_row = idxs_all_row.tolist()

    sims_row = sims_all_row.tolist()

    cand_ids = []

    for id_ in ids_row:

        if int(id_) == i:

            continue

        cand_ids.append(int(id_))

        if len(cand_ids) >= topN:

            break

    if len(cand_ids) == 0:

        return {
            "kmeans5-complexity": [],
            "kmeans5-quality": [],
            "kmeans5-compqual": [],
            "kmeans5-complexity-sims": [],
            "kmeans5-quality-sims": [],
            "kmeans5-compqual-sims": [],
        }

    X = emb[cand_ids]

    qual = np.array([data[j]["quality_score"] for j in cand_ids], dtype=np.float32)

    comp = np.array([data[j]["complexity_score"] for j in cand_ids], dtype=np.float32)

    n_clusters = min(k, len(cand_ids))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=km_n_init,
        max_iter=km_max_iter,
        algorithm="lloyd",
    )

    labels = kmeans.fit_predict(X)

    best_complex_ids, best_quality_ids, best_compqual_ids = [], [], []

    best_complex_sims, best_quality_sims, best_compqual_sims = [], [], []

    id2sim = {int(id_): float(sim) for id_, sim in zip(ids_row, sims_row)}

    for c_lab in range(n_clusters):

        idx_in_cluster = np.where(labels == c_lab)[0]

        if idx_in_cluster.size == 0:

            continue

        bc = idx_in_cluster[np.argmax(comp[idx_in_cluster])]

        bq = idx_in_cluster[np.argmax(qual[idx_in_cluster])]

        bsum = idx_in_cluster[np.argmax((comp + qual)[idx_in_cluster])]

        id_bc = int(cand_ids[bc])

        id_bq = int(cand_ids[bq])

        id_bsum = int(cand_ids[bsum])

        best_complex_ids.append(id_bc)

        best_quality_ids.append(id_bq)

        best_compqual_ids.append(id_bsum)

        best_complex_sims.append(id2sim.get(id_bc, float(np.dot(emb[id_bc], emb[i]))))

        best_quality_sims.append(id2sim.get(id_bq, float(np.dot(emb[id_bq], emb[i]))))

        best_compqual_sims.append(
            id2sim.get(id_bsum, float(np.dot(emb[id_bsum], emb[i])))
        )

    return {
        "kmeans5-complexity": best_complex_ids,
        "kmeans5-quality": best_quality_ids,
        "kmeans5-compqual": best_compqual_ids,
        "kmeans5-complexity-sims": best_complex_sims,
        "kmeans5-quality-sims": best_quality_sims,
        "kmeans5-compqual-sims": best_compqual_sims,
    }


def main():

    args = parse_args()

    np.random.seed(args.seed)

    data = load_data(args.input_path)

    emb = build_embeddings(
        data,
        sbert_model=args.sbert_model,
        device=args.device,
        batch_size=args.sbert_batch_size,
    )

    if args.save_embed_path:

        os.makedirs(os.path.dirname(args.save_embed_path) or ".", exist_ok=True)

        np.save(args.save_embed_path, emb)

    sims_all, idxs_all = faiss_search(
        emb, topN=max(args.top5, args.topN), use_gpu=args.faiss_gpu
    )

    for i, rec in enumerate(data):

        ids = idxs_all[i].tolist()

        sims = sims_all[i].tolist()

        try:

            self_pos = ids.index(i)

            ids.pop(self_pos)

            sims.pop(self_pos)

        except ValueError:

            pass

        ids = ids[: args.top5]

        sims = sims[: args.top5]

        rec["top5_similar_ids"] = [int(x) for x in ids]

        rec["top5_similar_sims"] = [float(x) for x in sims]

    print("Parallel KMeans on topN and per-cluster selections ...")

    results = Parallel(
        n_jobs=args.kmeans_jobs, prefer="threads", batch_size=args.kmeans_batch
    )(
        delayed(_per_item_kmeans)(
            i,
            data,
            emb,
            idxs_all[i],
            sims_all[i],
            args.topN,
            args.k,
            args.seed,
            args.kmeans_n_init,
            args.kmeans_max_iter,
        )
        for i in tqdm(range(len(data)), desc="Items")
    )

    for rec, out in zip(data, results):

        rec.update(out)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved → {args.output_path}")


if __name__ == "__main__":

    main()

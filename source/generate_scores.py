import argparse


import json


import os


import time


import numpy as np


import torch


import torch.distributed as dist


from scipy.special import softmax


from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():

    p = argparse.ArgumentParser(
        "Step 1 (slim): compute quality & complexity scores with DDP."
    )

    p.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSON (list of dicts with instruction,input,output)",
    )

    p.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write output JSON (only rank 0 writes)",
    )

    p.add_argument(
        "--quality_model", type=str, default="hkust-nlp/deita-quality-scorer"
    )

    p.add_argument(
        "--complexity_model", type=str, default="hkust-nlp/deita-complexity-scorer"
    )

    p.add_argument(
        "--max_len", type=int, default=512, help="Tokenize truncation length"
    )

    p.add_argument("--batch_size", type=int, default=16, help="Batch size per rank")

    return p.parse_args()


def setup_ddp():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank


def load_data(path):

    with open(path, "r", encoding="utf-8") as f:

        data = json.load(f)

    for idx, rec in enumerate(data):

        rec["id"] = idx

    return data


def save_data(path, data):

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def batch_infer(model, tokenizer, prompts, max_len, device, max_new_tokens=5):

    gen = model.module if hasattr(model, "module") else model

    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len
    ).to(device)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:

        enc["attention_mask"] = enc["attention_mask"]

        gen.config.pad_token_id = tokenizer.eos_token_id

    out = gen.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )

    return out.scores[0]


def score_batch_logits(logits, id2score):

    lp = logits.detach().cpu().numpy()

    idxs = list(id2score.keys())

    weights = np.array(list(id2score.values()), dtype=float)

    sub = lp[:, idxs]

    probs = softmax(sub, axis=1)

    return (probs * weights[None, :]).sum(axis=1)


def main():

    args = parse_args()

    rank = setup_ddp()

    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")

    t0 = time.time()

    data = load_data(args.input_path)

    if rank == 0:

        print(f"[Load] {len(data)} records in {time.time() - t0:.2f}s")

    t1 = time.time()

    q_tok = AutoTokenizer.from_pretrained(args.quality_model)

    q_mod = AutoModelForCausalLM.from_pretrained(args.quality_model).to(device)

    if world_size > 1:

        q_mod = torch.nn.parallel.DistributedDataParallel(q_mod, device_ids=[rank])

    q_mod.eval()

    q_prompts = [
        (
            rec.get("instruction", "")
            + ("\n" + rec.get("input", "") if rec.get("input") else "")
            + "\n#Response#:\n"
            + rec.get("output", "")
            + "\n##Quality:"
        )
        for rec in data
    ]

    local_idxs = list(range(rank, len(data), world_size))

    local_q = {}

    id2s = {29896: 1, 29906: 2, 29941: 3, 29946: 4, 29945: 5, 29953: 6}

    from tqdm import tqdm

    for i in tqdm(
        range(0, len(local_idxs), args.batch_size), desc=f"[Rank {rank}] Quality"
    ):

        batch_idxs = local_idxs[i : i + args.batch_size]

        batch_prompts = [q_prompts[j] for j in batch_idxs]

        logits = batch_infer(q_mod, q_tok, batch_prompts, args.max_len, device)

        scores = score_batch_logits(logits, id2s)

        for k, idx in enumerate(batch_idxs):

            local_q[idx] = float(scores[k])

    gathered_q = [None] * world_size

    dist.all_gather_object(gathered_q, local_q)

    if rank == 0:

        q_map = {}

        for d in gathered_q:

            q_map.update(d)

        for rec in data:

            rec["quality_score"] = q_map[rec["id"]]

        print(f"[Quality] done in {time.time() - t1:.2f}s")

    dist.barrier()

    del q_mod

    torch.cuda.empty_cache()

    t2 = time.time()

    c_tok = AutoTokenizer.from_pretrained(args.complexity_model)

    c_mod = AutoModelForCausalLM.from_pretrained(args.complexity_model).to(device)

    if world_size > 1:

        c_mod = torch.nn.parallel.DistributedDataParallel(c_mod, device_ids=[rank])

    c_mod.eval()

    c_prompts = [
        (
            rec.get("instruction", "")
            + ("\n" + rec.get("input", "") if rec.get("input") else "")
            + "\n##Complexity:"
        )
        for rec in data
    ]

    local_c = {}

    for i in tqdm(
        range(0, len(local_idxs), args.batch_size), desc=f"[Rank {rank}] Complexity"
    ):

        batch_idxs = local_idxs[i : i + args.batch_size]

        batch_prompts = [c_prompts[j] for j in batch_idxs]

        logits = batch_infer(c_mod, c_tok, batch_prompts, args.max_len, device)

        scores = score_batch_logits(logits, id2s)

        for k, idx in enumerate(batch_idxs):

            local_c[idx] = float(scores[k])

    gathered_c = [None] * world_size

    dist.all_gather_object(gathered_c, local_c)

    if rank == 0:

        c_map = {}

        for d in gathered_c:

            c_map.update(d)

        for rec in data:

            rec["complexity_score"] = c_map[rec["id"]]

        print(f"[Complexity] done in {time.time() - t2:.2f}s")

    dist.barrier()

    del c_mod

    torch.cuda.empty_cache()

    if rank == 0:

        save_data(args.output_path, data)

        print(f"[Save] {time.time() - t0:.2f}s → {args.output_path}")

    dist.destroy_process_group()


if __name__ == "__main__":

    main()

import os


import json


import argparse


import time


import torch


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from tqdm import tqdm


from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_DICT_NONE = {
    "prompt_input": "{instruction}\n{input}\n",
    "prompt_no_input": "{instruction}\n",
}


def get_perplexity_whole(tokenizer, model, text, max_length, device):

    try:

        ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():

            out = model(ids, labels=ids)

        return torch.exp(out.loss).item()

    except Exception:

        return 0.0


def get_perplexity_conditional(tokenizer, model, full_text, target, max_length, device):

    try:

        ids = tokenizer.encode(
            full_text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        decoded = tokenizer.decode(ids[0], skip_special_tokens=True)

        start = decoded.rfind(target)

        if start < 0:

            return 0.0

        prefix_ids = tokenizer.encode(decoded[:start], return_tensors="pt").to(device)

        split = prefix_ids.shape[1]

        labels = ids.clone()

        labels[0, :split] = -100

        with torch.no_grad():

            out = model(ids, labels=labels)

        return torch.exp(out.loss).item()

    except Exception:

        return 0.0


def parse_args():

    p = argparse.ArgumentParser(
        description="Step 2 (legacy): compute PPL/IFD and multi-list few_shot_ifd with DDP (no fs_ifd saved)"
    )

    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="输入 JSON，包含 id,instruction,input,output,top5_similar_ids,kmeans5-* 等字段",
    )

    p.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="输出 JSON：新增 ppl_A_direct, ppl_A_condition, ifd，以及多个 few_shot_ifd_* 列表（不再保存 fs_ifd_*）",
    )

    p.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="用于 PPL 计算的 CausalLM 模型（如 meta-llama/Llama-3.2-3B）",
    )

    p.add_argument(
        "--max_length", type=int, default=1024, help="单序列最大长度（右截断）"
    )

    return p.parse_args()


def setup_ddp():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return local_rank, dist.get_world_size()


def safe_get_ids(lst, N):

    out = []

    for x in lst or []:

        if isinstance(x, int) and 0 <= x < N:

            out.append(x)

    return out


def compute_few_ifd_for_list(tokenizer, model, device, A, data, cand_ids, max_length):

    few_ifd = []

    inp_A = (A.get("input") or "").strip()

    prompt_A = (
        PROMPT_DICT_NONE["prompt_input"]
        if inp_A
        else PROMPT_DICT_NONE["prompt_no_input"]
    ).format_map(A)

    out_A = A.get("output", "") or ""

    for sid in cand_ids:

        B = data[sid]

        inp_B = (B.get("input") or "").strip()

        prompt_B = (
            PROMPT_DICT_NONE["prompt_input"]
            if inp_B
            else PROMPT_DICT_NONE["prompt_no_input"]
        ).format_map(B)

        out_B = B.get("output", "") or ""

        few_ctx = prompt_A + out_A + prompt_B

        ppl_B_dir = get_perplexity_whole(tokenizer, model, out_B, max_length, device)

        ppl_B_cond = get_perplexity_conditional(
            tokenizer, model, few_ctx + out_B, out_B, max_length, device
        )

        if ppl_B_dir == 0.0 or ppl_B_cond == 0.0:

            few_ifd.append(0.0)

        else:

            few_ifd.append(ppl_B_cond / ppl_B_dir)

    return few_ifd


def main():

    args = parse_args()

    rank, world_size = setup_ddp()

    device = torch.device(f"cuda:{rank}")

    t0 = time.time()

    with open(args.data_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    for idx, rec in enumerate(data):

        rec.setdefault("id", idx)

    N = len(data)

    local_idxs = list(range(rank, N, world_size))

    if rank == 0:

        print(f"[Load] {N} records in {time.time() - t0:.2f}s")

    t1 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, output_hidden_states=False
    ).to(device)

    model = DDP(model, device_ids=[rank])

    model.eval()

    if rank == 0:

        print(f"[Model] loaded+DDP in {time.time() - t1:.2f}s")

    t2 = time.time()

    local_phase1 = {}

    for i in tqdm(local_idxs, desc=f"Rank {rank} Phase1"):

        A = data[i]

        inp = (A.get("input") or "").strip()

        prompt_A = (
            PROMPT_DICT_NONE["prompt_input"]
            if inp
            else PROMPT_DICT_NONE["prompt_no_input"]
        ).format_map(A)

        outp = A.get("output", "") or ""

        ppl_A_dir = get_perplexity_whole(
            tokenizer, model, outp, args.max_length, device
        )

        ppl_A_cond = get_perplexity_conditional(
            tokenizer, model, prompt_A + outp, outp, args.max_length, device
        )

        if ppl_A_dir == 0.0 or ppl_A_cond == 0.0:

            ifd = 0.0

        else:

            ifd = ppl_A_cond / ppl_A_dir

        local_phase1[i] = {
            "ppl_A_direct": ppl_A_dir,
            "ppl_A_condition": ppl_A_cond,
            "ifd": ifd,
        }

    gathered1 = [None] * world_size

    dist.all_gather_object(gathered1, local_phase1)

    if rank == 0:

        merged1 = {}

        for part in gathered1:

            merged1.update(part)

        for idx in range(N):

            if idx in merged1:

                data[idx].update(merged1[idx])

    dist.barrier()

    data_list = [data] if rank == 0 else [None]

    dist.broadcast_object_list(data_list, src=0)

    data = data_list[0]

    if rank == 0:

        print(f"[Phase1] done in {time.time() - t2:.2f}s")

    t3 = time.time()

    cand_specs = [
        ("top5_similar_ids", "few_shot_ifd_top5"),
        ("kmeans5-complexity", "few_shot_ifd_kmeans5_complexity"),
        ("kmeans5-quality", "few_shot_ifd_kmeans5_quality"),
        ("kmeans5-compqual", "few_shot_ifd_kmeans5_compqual"),
    ]

    local_phase2 = {}

    for i in tqdm(local_idxs, desc=f"Rank {rank} Phase2"):

        A = data[i]

        out_fields = {}

        for cand_key, few_key in cand_specs:

            cand_ids = safe_get_ids(A.get(cand_key, []) or [], N)

            if not cand_ids:

                out_fields[few_key] = []

                continue

            few_ifd_list = compute_few_ifd_for_list(
                tokenizer, model, device, A, data, cand_ids, args.max_length
            )

            out_fields[few_key] = few_ifd_list

        local_phase2[i] = out_fields

    gathered2 = [None] * world_size

    dist.all_gather_object(gathered2, local_phase2)

    if rank == 0:

        merged2 = {}

        for part in gathered2:

            if part:

                merged2.update(part)

        for idx in range(N):

            if idx in merged2:

                data[idx].update(merged2[idx])

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

        with open(args.save_path, "w", encoding="utf-8") as fw:

            json.dump(data, fw, ensure_ascii=False, indent=2)

        print(f"[Save] -> {args.save_path} | Phase2 in {time.time() - t3:.2f}s")

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":

    main()

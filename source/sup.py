import json


import argparse


from tqdm import tqdm


import math


def recompute_fs_scores(data):

    id2rec = {rec["id"]: rec for rec in data}

    for rec in tqdm(data, desc="Recomputing fs_score"):

        sims = rec.get("top5_similar_ids", [])

        few_ifds = rec.get("few_shot_ifd", [])

        new_fs = []

        for sid, fs_val in zip(sims, few_ifds):

            orig_ifd = id2rec[sid].get("ifd", float("inf"))

            if orig_ifd > 0 and not math.isinf(orig_ifd):

                new_fs.append((orig_ifd - fs_val) / orig_ifd)

            else:

                new_fs.append(0.0)

        rec["fs_score"] = new_fs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Recompute fs_score for a whole JSON file"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/alpaca_data-step2.json",
        help="输入 JSON 文件路径，格式为 list of dicts",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/alpaca_data-step2.5.json",
        help="输出 JSON 文件路径",
    )

    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    recompute_fs_scores(data)

    with open(args.output_path, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Recomputed fs_score for {len(data)} records → saved to {args.output_path}")

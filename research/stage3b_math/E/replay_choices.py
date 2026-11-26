"""Paired, fixed-checkpoint prompt replay on unrotated Stage 3 test mothers.

Pure helpers are importable without model dependencies. For GPU usage, see E_host.md.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import unicodedata


BASE = "69fbd29acdb4eba7961d56176693786b6181477a"
ENCODER = "equivariant_b8_s17"
CONDITIONS = ("C1_coherent_2db", "C4_cfo_ramp")
INSTRUCTION = "首行只写所选答案的完整选项文字；第二行可用一句中文说明判断。"


def prompts(question, choices):
    if len(choices) != 4 or len(set(choices)) != 4:
        raise ValueError("Expected four distinct candidate strings")
    listed = "\n".join(choices)
    return {
        "omitted": f"{question}\n{INSTRUCTION}",
        "listed": f"{question}\n四个选项如下。\n{listed}\n{INSTRUCTION}",
    }


def parse_choice(response, choices):
    """Accept a canonical first line, with optional surrounding Markdown emphasis."""
    normalize = lambda x: unicodedata.normalize("NFKC", x.strip().strip("*`").strip())
    first = response.strip().splitlines()
    if not first:
        return None
    observed = normalize(first[0])
    return next((i for i, choice in enumerate(choices) if normalize(choice) == observed), None)


def summarize(rows):
    result = defaultdict(lambda: defaultdict(int))
    for row in rows:
        truth = row["true_index"]
        group = result[row["condition"] + "/" + row["task"]]
        group["n"] += 1
        group["original_score_correct"] += int(row["original_score"] == truth)
        for arm in ("omitted", "listed"):
            generated = row[arm]["generated_index"]
            scored = row[arm]["score_index"]
            group[arm + "_parseable"] += int(generated is not None)
            group[arm + "_generation_correct"] += int(generated == truth)
            group[arm + "_candidate_correct"] += int(scored == truth)
            group[arm + "_agreement"] += int(generated == scored)
        old = row["omitted"]["generated_index"] == truth
        new = row["listed"]["generated_index"] == truth
        group["generation_corrected"] += int(new and not old)
        group["generation_new_errors"] += int(old and not new)
    return {key: dict(value) for key, value in result.items()}


def replay(repo, count):
    import torch
    from huggingface_hub import snapshot_download

    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", BASE, "--",
                    "research/stage3_ministral_cache", "research/matched_cache"], check=True)
    sys.path.insert(0, str(repo.resolve()))
    from research.stage3_ministral_cache import chat_demo as baseline
    from research.stage3_ministral_cache.schema import build_query_prompt

    source = Path(baseline.__file__).resolve().parent
    lock = json.loads((source / "results/host/host_dev_lock.json").read_text(encoding="utf-8"))
    ck = lock["encoders"][ENCODER]
    weights = source / "connector_weights" / f"connector_{ENCODER}.pt"
    if sha256(weights.read_bytes()).hexdigest() != ck["checkpoint_sha256"]:
        raise ValueError("Connector checksum mismatch")
    local = snapshot_download(lock["model_id"], revision=lock["hf_sha"])
    host = baseline.MinistralCacheHost(model_id=local, dtype=torch.bfloat16)
    if host.host_sha256 != lock["host_sha256"]:
        raise ValueError("Frozen host fingerprint does not match the historical BF16 run")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    connector = baseline.WaveformCacheConnector(**state["connector_config"]).to(host.device).eval()
    connector.load_state_dict(state["state_dict"])
    encoder = baseline.load_frozen_models()[ENCODER]
    temperature = ck["locked_temp"]
    rows = []
    with torch.no_grad():
        for condition in CONDITIONS:
            # The generator uses block count in its RNG consumption. Always reproduce the
            # two-block host batch even for a short replay; only slice AFTER generation.
            batch = baseline.generate_mothers("s3_test", condition, max_blocks=lock["host_test_blocks_locked"])
            z1, z2 = encoder.encode(batch.y1), encoder.encode(batch.y2)
            refs = batch.references_for_arm("valid")
            ref_features = [torch.as_tensor(baseline.build_ref_feature(ref)) for ref in refs]
            q1 = build_query_prompt("Q1", batch.condition)
            q3 = build_query_prompt("Q3", batch.condition, protocol="repeat_x1")
            q2 = build_query_prompt("Q2", batch.condition, ref_arm="valid", references=refs)
            for index in range(count):
                codes = [torch.tensor([int(z[index])], dtype=torch.long) for z in (z1, z2)]
                for task, (question, choices), features, truth in (
                    ("Q1", q1, [], (batch.x2[index] - batch.x1[index]) % 4),
                    ("Q3_repeat", q3, [], batch.k3_repeat[index]),
                    ("Q2_valid", q2, [x[index:index + 1] for x in ref_features], batch.x1[index]),
                ):
                    row = {"condition": condition, "mother_id": batch.mother_ids[index],
                           "task": task, "choices": list(choices), "true_index": int(truth)}
                    variants = {"original": question, **prompts(question, choices)}
                    for arm, prompt in variants.items():
                        prefix = host.build_prefix_embeds(connector, prompt, *codes, ref_feats_list=features)
                        log_probs, _ = host.score_candidates_kv(prefix, choices, temperature=temperature)
                        score = int(log_probs[0].argmax().item())
                        if arm == "original":
                            row["original_score"] = score
                            continue
                        reply = host.generate_text(prefix, max_new_tokens=96)
                        row[arm] = {"score_index": score, "reply": reply,
                                    "generated_index": parse_choice(reply, choices)}
                    rows.append(row)
    return {"source_commit": BASE, "hf_revision": lock["hf_sha"],
            "host_fingerprint": host.host_sha256, "connector_sha256": ck["checkpoint_sha256"],
            "temperature": temperature, "mothers_per_condition": count, "rotations": [[0, 0]],
            "summary": summarize(rows), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Checkout containing frozen Stage 3 sources")
    parser.add_argument("--mothers", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.mothers <= 128:
        parser.error("--mothers must be between 1 and 128")
    output = replay(args.repo, args.mothers)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

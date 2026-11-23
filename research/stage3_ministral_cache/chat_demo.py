"""Step C.4 Replayable Chinese multi-turn Q&A transcript on held-out s3_test using locked 3B host."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from research.stage3_ministral_cache.analytic import readout_from_table
from research.stage3_ministral_cache.data import generate_mothers
from research.stage3_ministral_cache.eval_direct import load_frozen_models
from research.stage3_ministral_cache.eval_host import _unique_ref_feats
from research.stage3_ministral_cache.ministral_connector import (
    MinistralCacheHost,
    WaveformCacheConnector,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    Q1_CHOICES,
    Q2_CHOICES,
    Q3_CHOICES,
    build_query_prompt,
)


STAGE3_DIR = Path(__file__).resolve().parent


def _score_and_generate_single(
    host: MinistralCacheHost,
    connector: WaveformCacheConnector,
    prompt: str,
    choices,
    z1_val: int,
    z2_val: int,
    r_feats_single: List[np.ndarray],
    cal_temp: float,
    ask_explain: bool = True,
) -> dict:
    z1_t = torch.tensor([z1_val], dtype=torch.long)
    z2_t = torch.tensor([z2_val], dtype=torch.long)
    r_t = [torch.tensor(rf[None, :], dtype=torch.float32) for rf in r_feats_single]
    with torch.no_grad():
        pref = host.build_prefix_embeds(connector, prompt, z1_t, z2_t, ref_feats_list=r_t)
        lp, raw_ll = host.score_candidates_kv(pref, choices, temperature=cal_temp)
        probs = lp.exp()[0].detach().cpu().numpy().tolist()

        if ask_explain:
            explain_prompt = f"{prompt}\n请从以上四个选项中明确选出一项，并用一句简短中文说明判断。"
        else:
            explain_prompt = prompt
        pref_gen = host.build_prefix_embeds(connector, explain_prompt, z1_t, z2_t, ref_feats_list=r_t)
        gen_reply = host.generate_text(pref_gen, max_new_tokens=96)

    top_idx = int(np.argmax(probs))
    return {
        "prompt": prompt,
        "choices": list(choices),
        "probs": [round(float(p), 5) for p in probs],
        "pred_idx": top_idx,
        "pred_choice": choices[top_idx],
        "generated_reply_zh": gen_reply,
    }


def _select_mother_index(enc, batch) -> int:
    """Pick first mother in block where valid pilot helps or updates posterior strongly vs none."""
    z1 = np.asarray(enc.encode(batch.y1), dtype=np.int64)
    z2 = np.asarray(enc.encode(batch.y2), dtype=np.int64)
    a1, a2 = enc.table[z1], enc.table[z2]
    p_none = readout_from_table(a1, a2, (), batch.condition)["q2"]
    p_val = readout_from_table(a1, a2, batch.references_for_arm("valid"), batch.condition)["q2"]
    for i in range(len(batch.x1)):
        if int(np.argmax(p_none[i])) != int(batch.x1[i]) and int(np.argmax(p_val[i])) == int(batch.x1[i]):
            return i
    return 0


def run_chat_transcript(out_dir: Path = STAGE3_DIR, enc_name: str = "equivariant_b8_s17") -> dict:
    res_dir = out_dir / "results" / "host"
    lock_path = res_dir / "host_dev_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing {lock_path}")
    dev_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    enc_lock = dev_lock["encoders"][enc_name]
    cal_temp = float(enc_lock["locked_temp"])

    host = MinistralCacheHost(model_id=dev_lock["model_id"])
    assert host.host_sha256 == dev_lock["host_sha256"]

    models = load_frozen_models()
    enc = models[enc_name]
    ckpt = torch.load(out_dir / "connector_weights" / f"connector_{enc_name}.pt", map_location="cpu", weights_only=True)
    conn_cfg = ckpt.get(
        "connector_config",
        {
            "num_states": int(ckpt["num_states"]),
            "hidden_size": int(ckpt["hidden_size"]),
            "target_embed_std": float(ckpt["target_embed_std"]),
        },
    )
    connector = WaveformCacheConnector(**conn_cfg).to(host.device)
    connector.load_state_dict(ckpt["state_dict"])
    connector.eval()

    cases_out = []
    md_lines = [
        "# Ministral-3-3B 锁参中文多轮交互与信念修正实录",
        "",
        f"- 冻结宿主：`{dev_lock['model_id']}` (HF SHA `{host.hf_sha}`)",
        f"- 冻结编码器：`{enc_name}`（宿主严格只见整数缓存 `(z1, z2)` 与显式揭示导频，不见原始 `Y1/Y2` 或真值标签）",
        f"- 温度标定（锁定自 `s3_dev`）：`T = {cal_temp:.2f}`",
        "",
    ]

    for cname in ("C1_coherent_2db", "C3_mid_pilot_cont"):
        cond = CONDITION_SPECS[cname]
        batch = generate_mothers("s3_test", cname, max_blocks=1)
        idx = _select_mother_index(enc, batch)
        z1_i = int(enc.encode(batch.y1[idx : idx + 1])[0])
        z2_i = int(enc.encode(batch.y2[idx : idx + 1])[0])
        m_id = batch.mother_ids[idx]
        x1_true = int(batch.x1[idx])
        x2_true = int(batch.x2[idx])
        q1_true = int((x2_true - x1_true) % 4)
        k3_rep_true = int(batch.k3_repeat[idx])
        k3_ind_true = int(batch.k3_indep[idx])

        # Gather single-mother Reference tuples
        refs_valid = tuple(
            type(r)(r.acquisition_id, r.shares_channel, r.samples[idx], r.t_p, r.snr_db)
            for r in batch.references_for_arm("valid")
        )
        refs_indep = tuple(
            type(r)(r.acquisition_id, r.shares_channel, r.samples[idx], r.t_p, r.snr_db)
            for r in batch.references_for_arm("independent")
        )
        refs_dup = tuple(
            type(r)(r.acquisition_id, r.shares_channel, r.samples[idx], r.t_p, r.snr_db)
            for r in batch.references_for_arm("duplicate_copy")
        )
        refs_fresh = tuple(
            type(r)(r.acquisition_id, r.shares_channel, r.samples[idx], r.t_p, r.snr_db)
            for r in batch.references_for_arm("fresh_resample")
        )

        # Turns
        p_q1, ch_q1 = build_query_prompt("Q1", cond)
        t1 = _score_and_generate_single(host, connector, p_q1, ch_q1, z1_i, z2_i, [], cal_temp)
        t1["turn"] = "Turn 1: 揭示前相对关系 (Q1)"
        t1["true_label"] = Q1_CHOICES[q1_true]

        p_q3r, ch_q3r = build_query_prompt("Q3", cond, protocol="repeat_x1")
        t2_rep = _score_and_generate_single(host, connector, p_q3r, ch_q3r, z1_i, z2_i, [], cal_temp)
        t2_rep["turn"] = "Turn 2A: 揭示前第三窗预测 (Q3 repeat_x1)"
        t2_rep["true_label"] = Q3_CHOICES[k3_rep_true]

        p_q3i, ch_q3i = build_query_prompt("Q3", cond, protocol="independent_x3")
        t2_ind = _score_and_generate_single(host, connector, p_q3i, ch_q3i, z1_i, z2_i, [], cal_temp)
        t2_ind["turn"] = "Turn 2B: 揭示前第三窗控制 (Q3 independent_x3)"
        t2_ind["true_label"] = Q3_CHOICES[k3_ind_true]

        p_q2_n, ch_q2 = build_query_prompt("Q2", cond, ref_arm="none", references=())
        t3_none = _score_and_generate_single(host, connector, p_q2_n, ch_q2, z1_i, z2_i, [], cal_temp)
        t3_none["turn"] = "Turn 3A: 无参考绝对位置 (Q2 none)"
        t3_none["true_label"] = Q2_CHOICES[x1_true]

        p_q2_i, _ = build_query_prompt("Q2", cond, ref_arm="independent", references=refs_indep, deduplicate=True)
        rf_i = [build_ref_feature(r) for r in refs_indep]
        t3_ind = _score_and_generate_single(host, connector, p_q2_i, ch_q2, z1_i, z2_i, rf_i, cal_temp, ask_explain=False)
        t3_ind["turn"] = "Turn 3B: 揭示独立无关参考 (Q2 independent)"
        t3_ind["true_label"] = Q2_CHOICES[x1_true]

        p_q2_v, _ = build_query_prompt("Q2", cond, ref_arm="valid", references=refs_valid, deduplicate=True)
        rf_v = [build_ref_feature(r) for r in refs_valid]
        t4_val = _score_and_generate_single(host, connector, p_q2_v, ch_q2, z1_i, z2_i, rf_v, cal_temp)
        t4_val["turn"] = "Turn 4A: 揭示有效导频后修正 (Q2 valid)"
        t4_val["true_label"] = Q2_CHOICES[x1_true]

        p_q3a, ch_q3a = build_query_prompt("Q3_after_pilot", cond, protocol="repeat_x1", ref_arm="valid", references=refs_valid, deduplicate=True)
        t4_q3a = _score_and_generate_single(host, connector, p_q3a, ch_q3a, z1_i, z2_i, rf_v, cal_temp)
        t4_q3a["turn"] = "Turn 4B: 导频修正后第三窗更新 (Q3_after_pilot)"
        t4_q3a["true_label"] = Q3_CHOICES[k3_rep_true]

        p_q2_d, _ = build_query_prompt("Q2", cond, ref_arm="duplicate_copy", references=refs_dup, deduplicate=True)
        rf_d = [build_ref_feature(refs_dup[0])]  # deduped
        t5_dup = _score_and_generate_single(host, connector, p_q2_d, ch_q2, z1_i, z2_i, rf_d, cal_temp, ask_explain=False)
        t5_dup["turn"] = "Turn 5A: 同一采集编号重复拷贝（共享去重契约 duplicate_copy）"
        t5_dup["max_abs_diff_vs_valid"] = float(np.max(np.abs(np.array(t5_dup["probs"]) - np.array(t4_val["probs"]))))

        p_q2_rd, _ = build_query_prompt("Q2", cond, ref_arm="duplicate_copy", references=refs_dup, deduplicate=False)
        rf_rd = [build_ref_feature(r) for r in refs_dup]
        t5_raw_dup = _score_and_generate_single(host, connector, p_q2_rd, ch_q2, z1_i, z2_i, rf_rd, cal_temp, ask_explain=False)
        t5_raw_dup["turn"] = "Turn 5B: 未去重重复令牌探针 (raw_duplicate_probe)"
        t5_raw_dup["tv_vs_valid"] = float(0.5 * np.sum(np.abs(np.array(t5_raw_dup["probs"]) - np.array(t4_val["probs"]))))

        p_q2_f, _ = build_query_prompt("Q2", cond, ref_arm="fresh_resample", references=refs_fresh, deduplicate=True)
        rf_f = [build_ref_feature(r) for r in refs_fresh]
        t5_fresh = _score_and_generate_single(host, connector, p_q2_f, ch_q2, z1_i, z2_i, rf_f, cal_temp)
        t5_fresh["turn"] = "Turn 5C: 新鲜独立再采样双导频 (Q2 fresh_resample)"
        t5_fresh["true_label"] = Q2_CHOICES[x1_true]

        turns = [t1, t2_rep, t2_ind, t3_none, t3_ind, t4_val, t4_q3a, t5_dup, t5_raw_dup, t5_fresh]
        cases_out.append(
            {
                "condition": cname,
                "mother_id": m_id,
                "z1": z1_i,
                "z2": z2_i,
                "x1_true": x1_true,
                "x2_true": x2_true,
                "turns": turns,
            }
        )

        md_lines.append(f"## 工况 `{cname}` — 母样 `{m_id}` (整数缓存 `z1={z1_i}, z2={z2_i}`)")
        md_lines.append("")
        md_lines.append(f"> 真值对照（仅评测离线核对，未输入模型）：`X1={Q2_CHOICES[x1_true]}`, `X2={Q2_CHOICES[x2_true]}`, `Q1={Q1_CHOICES[q1_true]}`, `Q3(repeat)={Q3_CHOICES[k3_rep_true]}`")
        md_lines.append("")
        for t in turns:
            md_lines.append(f"### {t['turn']}")
            md_lines.append(f"- **输入提示**: `{t['prompt'].replace(chr(10), ' / ')}`")
            md_lines.append(f"- **4选校准概率**: `{t['probs']}` -> 首选 `{t['pred_choice']}`" + (f" (真值: `{t['true_label']}`)" if "true_label" in t else ""))
            if "max_abs_diff_vs_valid" in t:
                md_lines.append(f"- **相对 valid 最大概率偏差**: `{t['max_abs_diff_vs_valid']:.3e}`")
            if "tv_vs_valid" in t:
                md_lines.append(f"- **相对 valid 总变差 (TV)**: `{t['tv_vs_valid']:.5f}`")
            md_lines.append(f"- **模型中文作答**: {t['generated_reply_zh']}")
            md_lines.append("")

    (res_dir / "transcript_zh.md").write_text("\n".join(md_lines), encoding="utf-8")
    payload = {"model_id": dev_lock["model_id"], "hf_sha": host.hf_sha, "encoder": enc_name, "cal_temp": cal_temp, "cases": cases_out}
    (res_dir / "transcript_zh.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved Chinese replay transcript to {res_dir / 'transcript_zh.md'}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="equivariant_b8_s17")
    args = ap.parse_args()
    run_chat_transcript(enc_name=args.encoder)

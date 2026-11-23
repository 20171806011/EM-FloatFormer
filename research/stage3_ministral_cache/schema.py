"""Shared schemas, canonical QPSK sectors, prompts, and reference deduplication."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


ROOTS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
N_SAMPLES = 3
TWO_DB_SNR = 10.0 ** 0.2
ROTATIONS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1))
ARMS: Tuple[str, ...] = ("none", "valid", "independent", "duplicate_copy", "fresh_resample")
PROTOCOLS: Tuple[str, ...] = ("repeat_x1", "independent_x3")

Q1_PROMPT = "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
Q1_CHOICES: Tuple[str, ...] = ("0圈（0°）", "1圈（+90°）", "2圈（180°）", "3圈（+270°）")

Q2_PROMPT = "第一个窗口原来发送的是哪个四相符号？"
Q2_CHOICES: Tuple[str, ...] = ("符号0（1）", "符号1（i）", "符号2（-1）", "符号3（-i）")

Q3_PROMPT_REPEAT = (
    "协议规定第三个窗口重发第一个窗口的符号，且已知当前时间间隔。"
    "在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区？"
)
Q3_PROMPT_INDEP = (
    "协议规定第三个窗口发送与前两窗独立均匀的新四相符号。"
    "在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区？"
)
Q3_PROMPT_REPEAT_AFTER_PILOT = (
    "已揭示上述参考导频；协议规定第三个窗口重发第一个窗口的符号。"
    "综合原缓存与导频后，第三个窗口接收相位最可能落入哪个四分扇区？"
)
Q3_CHOICES: Tuple[str, ...] = (
    "扇区0（-45°到+45°）",
    "扇区1（+45°到+135°）",
    "扇区2（+135°到-135°）",
    "扇区3（-135°到-45°）",
)


@dataclass(frozen=True)
class Reference:
    """Reference observation with explicit acquisition ID and channel-sharing declaration."""
    acquisition_id: int
    shares_channel: bool
    samples: np.ndarray  # shape (..., 3) or (3,) complex128
    t_p: float = 0.0
    snr_db: float = 2.0


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    snr_db: float
    h: int
    ell: int
    t_p: float
    drift_type: str  # "coherent" | "discrete" | "continuous" | "cfo_continuous"
    d: float = 0.0
    sigma_phi: float = 0.0
    omega: float = 0.0
    in_support_encoder: bool = False
    in_support_connector: bool = False
    question_order: str = "standard"

    @property
    def snr(self) -> float:
        return 10.0 ** (self.snr_db / 10.0)


CONDITION_SPECS: Dict[str, ConditionSpec] = {
    "C1_coherent_2db": ConditionSpec(
        name="C1_coherent_2db",
        snr_db=2.0,
        h=0,
        ell=0,
        t_p=0.0,
        drift_type="coherent",
        in_support_encoder=True,
        in_support_connector=True,
    ),
    "C2_discrete_walk": ConditionSpec(
        name="C2_discrete_walk",
        snr_db=2.0,
        h=2,
        ell=2,
        t_p=6.0,
        drift_type="discrete",
        d=0.08,
        in_support_encoder=False,
        in_support_connector=True,
    ),
    "C3_mid_pilot_cont": ConditionSpec(
        name="C3_mid_pilot_cont",
        snr_db=2.0,
        h=4,
        ell=2,
        t_p=2.0,
        drift_type="continuous",
        sigma_phi=0.25,
        in_support_encoder=False,
        in_support_connector=True,
    ),
    "C4_cfo_ramp": ConditionSpec(
        name="C4_cfo_ramp",
        snr_db=2.0,
        h=2,
        ell=2,
        t_p=4.0,
        drift_type="cfo_continuous",
        sigma_phi=0.10,
        omega=0.35,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C5_snr_0db": ConditionSpec(
        name="C5_snr_0db",
        snr_db=0.0,
        h=1,
        ell=2,
        t_p=3.0,
        drift_type="continuous",
        sigma_phi=0.20,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C5_snr_6db": ConditionSpec(
        name="C5_snr_6db",
        snr_db=6.0,
        h=1,
        ell=2,
        t_p=3.0,
        drift_type="continuous",
        sigma_phi=0.20,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C6_decoherent_order": ConditionSpec(
        name="C6_decoherent_order",
        snr_db=2.0,
        h=2,
        ell=2,
        t_p=4.0,
        drift_type="continuous",
        sigma_phi=3.0,
        in_support_encoder=False,
        in_support_connector=False,
        question_order="unseen_q3_first",
    ),
}


def sector_of_complex(z: np.ndarray) -> np.ndarray:
    """Canonical half-open QPSK sectors matching research/matched_cache/learned.py."""
    z_arr = np.asarray(z, dtype=np.complex128)
    u = z_arr.real + z_arr.imag
    v = z_arr.imag - z_arr.real
    out = np.zeros(z_arr.shape, dtype=np.int64)
    out[(u > 0.0) & (v >= 0.0)] = 1
    out[(u <= 0.0) & (v > 0.0)] = 2
    out[(u < 0.0) & (v <= 0.0)] = 3
    return out


def matched_sum(windows: np.ndarray) -> np.ndarray:
    """Orthogonal matched-filter statistic T = sum(y) / sqrt(3)."""
    arr = np.asarray(windows, dtype=np.complex128)
    return arr.sum(axis=-1) / np.sqrt(float(N_SAMPLES))


def dedup_references(references: Sequence[Reference]) -> Tuple[List[Reference], int, int, bool]:
    """Shared deduplication contract across direct readout and 3B host.

    Groups references by `(acquisition_id, shares_channel == True)` and verifies that
    identical acquisition_id entries do not point to conflicting samples.
    """
    seen_all: Dict[int, Reference] = {}
    unique_valid: List[Reference] = []
    raw_valid_count = 0
    for ref in references:
        samples = np.asarray(ref.samples, dtype=np.complex128)
        if ref.acquisition_id in seen_all:
            prev = seen_all[ref.acquisition_id]
            if bool(prev.shares_channel) != bool(ref.shares_channel) or abs(prev.t_p - ref.t_p) > 1e-12:
                raise ValueError(f"Conflicting metadata for acquisition_id={ref.acquisition_id}")
            if not np.array_equal(samples, np.asarray(prev.samples, dtype=np.complex128)):
                raise ValueError(f"Conflicting samples for acquisition_id={ref.acquisition_id}")
            if ref.shares_channel:
                raw_valid_count += 1
            continue
        seen_all[ref.acquisition_id] = ref
        if ref.shares_channel:
            raw_valid_count += 1
            unique_valid.append(ref)
    dedup_applied = raw_valid_count != len(unique_valid)
    return unique_valid, raw_valid_count, len(unique_valid), dedup_applied


def build_query_prompt(
    question_type: str,
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    ref_arm: str = "none",
    references: Sequence[Reference] = (),
    include_ref_token: bool = True,
    deduplicate: bool = True,
) -> Tuple[str, Tuple[str, ...]]:
    """Construct exact Chinese prompt (only public metadata + modality placeholder tokens).

    Placeholder tokens: `<W1>`, `<W2>` always present; `<REF>` present only when references
    are supplied (and if `deduplicate=False` for raw_duplicate_probe, emits one `<REF>` per
    raw reference entry so host-side sensitivity to duplicate tokens can be measured).
    """
    if question_type not in ("Q1", "Q2", "Q3", "Q3_after_pilot"):
        raise ValueError(f"Unknown question_type={question_type}")
    drift_desc_map = {
        "coherent": "相干静止(漂移=0)",
        "discrete": f"离散C4游走(d={cond.d:.2f}/步)",
        "continuous": f"连续高斯游走(sigma={cond.sigma_phi:.2f}rad/步)",
        "cfo_continuous": f"线性频偏+游走(omega={cond.omega:.2f},sigma={cond.sigma_phi:.2f})",
    }
    meta = (
        f"工况={cond.name}; 信噪比={cond.snr_db:.1f}dB; "
        f"两窗间隔h={cond.h}, 留出间隔ell={cond.ell}, 导频时刻tp={cond.t_p:.1f}; "
        f"信道={drift_desc_map[cond.drift_type]}。\n"
        f"已编码波形：<W1><W2>。"
    )
    if question_type == "Q1":
        return f"{meta}\n问题：{Q1_PROMPT}", Q1_CHOICES
    if question_type == "Q3":
        qtext = Q3_PROMPT_REPEAT if protocol == "repeat_x1" else Q3_PROMPT_INDEP
        return f"{meta}\n问题：{qtext}", Q3_CHOICES

    # Q2 or Q3_after_pilot (post reference reveal)
    if len(references) == 0:
        ref_text = "补充参考：无参考观测。"
    else:
        if deduplicate:
            seen_ids = set()
            items = []
            for r in references:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                tag = "同信道" if r.shares_channel else "独立无关"
                tok = "<REF>" if include_ref_token else ""
                items.append(f"[编号={r.acquisition_id},状态={tag},时刻={r.t_p:.1f}]{tok}")
            dup_note = f"（已按唯一采集编号去重，原始共{len(references)}条，去重后{len(items)}条）"
            ref_text = "补充参考：" + "；".join(items) + dup_note + "。"
        else:
            items = []
            for r in references:
                tag = "同信道" if r.shares_channel else "独立无关"
                tok = "<REF>" if include_ref_token else ""
                items.append(f"[编号={r.acquisition_id},状态={tag},时刻={r.t_p:.1f}]{tok}")
            ref_text = f"补充参考（未去重，共{len(items)}条）：" + "；".join(items) + "。"

    qtext = Q2_PROMPT if question_type == "Q2" else Q3_PROMPT_REPEAT_AFTER_PILOT
    choices = Q2_CHOICES if question_type == "Q2" else Q3_CHOICES
    return f"{meta}\n{ref_text}\n问题：{qtext}", choices

"""S03 ambiguity audit. Keep payload sign errors and carrier quadrant slips as separate endpoints."""
import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from p4.em_floatformer import compute_gauge_anchor
from p4.next_neural import ARMS, digest
from p4.next_pilots import bit_signs, fft_after_cfo, pilot_refined_cfo, legacy_static_cfo
from p4.slot_2d import EMFloatFormer2D, SlotSimulator2D

FOLDER = Path(__file__).resolve().parent / "results"
S, N, CP = 14, 32, 16
PUBLIC_SEED = 42
DEV_SEEDS = tuple(range(837101, 837109))


def receive(rx, pilots, mask, nv, model, epsilon=None):
    """Production path needs only received samples, known pilots, mask, noise variance, and weights."""
    num_symbols, n = mask.shape
    cp = rx.size(1) // num_symbols - n
    if epsilon is None:
        _, epsilon = pilot_refined_cfo(rx, pilots, mask, cp, tap_count=4)
    y = fft_after_cfo(rx, epsilon, num_symbols, n, cp)
    yp = y[:, mask] * pilots.conj()
    internal = {}
    def record(key):
        def hook(_module, _inputs, output):
            internal[key] = output.squeeze(-1)
        return hook
    with ExitStack() as ctx:
        ctx.enter_context(model.gamma_eff.register_forward_hook(record("effective_channel")))
        ctx.enter_context(model.head_dist.register_forward_hook(record("distortion")))
        llr = model(yp, y, mask, nv).llrs
    h, dist = internal["effective_channel"], internal["distortion"]
    anchor = compute_gauge_anchor(yp.to(torch.complex128),eps=1e-38).unsqueeze(-1)
    z = ((y.to(torch.complex128)/anchor - dist) * h.conj()) / (h.abs().square()+1e-4)
    if not torch.equal(bit_signs(z), llr[..., [0,2]] > 0):
        raise ValueError("Extracted constellation signs disagree with the public receiver")
    return llr, z, epsilon


def phase_and_errors(z, data, data_mask):
    """Transmitted data enter the offline scorer after receiver predictions are fixed."""
    mismatch = bit_signs(z[:,data_mask]) != bit_signs(data)
    products = torch.zeros_like(z)
    products[:,data_mask] = z[:,data_mask] * data.conj()
    row_sum = products.sum(-1)
    # A zero correlation has no defined phase. Use a relative cancellation
    # tolerance so changing the units of the received data preserves observability.
    observable = row_sum.abs() > (32*torch.finfo(z.real.dtype).eps * products.abs().sum(-1))
    signed_row_phase = row_sum.angle()
    quadrant_index = torch.round(signed_row_phase / (.5 * math.pi)).long().remainder(4)
    return {
        "sign_errors":int(mismatch.sum()), "sign_total":mismatch.numel(),
        "wrong_symbol_quadrants":int(mismatch.any(-1).sum()), "data_symbols":data.numel(),
        "slot_mean_abs_phase_deg":products.sum((-1,-2)).angle().abs().rad2deg().tolist(),
        "row_abs_phase_deg":signed_row_phase.abs().rad2deg().tolist(),
          "wrong_carrier_quadrant_rows":int(((quadrant_index!=0) & observable).sum()),
          "unobservable_rows":int((~observable).sum()),
          "carrier_quadrant_transitions":int(((quadrant_index[:,1:] != quadrant_index[:,:-1])
                                               & observable[:,1:] & observable[:,:-1]).sum()),
        "rows":quadrant_index.numel(), "adjacent_row_pairs":quadrant_index[:,1:].numel(),
    }


def full_wave_genie(slot):
    """Single-bit genie and disjoint-symbol all-correct bound for uncoded clean complex AWGN."""
    if slot.A is not None or slot.alpha != 0.0 or slot.info_bits is not None:
        raise ValueError("Genie model requires a linear PA and independent uncoded payload")
    length = N + CP
    b = len(slot.rx_time)
    gains = slot.h_time_evol.to(torch.complex128)
    n_taps = gains.size(-1)
    local = torch.arange(length+n_taps-1, dtype=torch.float64)
    time = length*torch.arange(S,dtype=torch.float64)[:,None] + local
    valid = time < S*length
    tone = torch.arange(N,dtype=torch.float64)
    # Each tap in this simulator is one complex sinusoid. Two true midpoints
    # determine its angular frequency unambiguously for these fd values.
    omega = (gains[:,1] * gains[:,0].conj()).angle() / length
    mids = CP + N/2
    columns = torch.zeros((b,S,len(local),N),dtype=torch.complex128)
    cfo_rotation = torch.exp(2j*math.pi*slot.cfo*time/N)
    for l in range(n_taps):
        h = gains[:,0,l,None,None]*torch.exp(1j*omega[:,l,None,None]*(time-mids))
        tx = torch.exp(2j*math.pi*(local[:,None]-l-CP)*tone/N) / math.sqrt(N)
        tx = tx * ((local>=l)&(local<length+l))[:,None]
        columns += (h*cfo_rotation*valid)[...,None]*tx
    noiseless = torch.zeros((b,S*length),dtype=torch.complex128)
    contributions = (columns*slot.x_grid[:,:,None,:]).sum(-1)
    for s in range(S):
        count = min(len(local),(S-s)*length)
        noiseless[:,s*length:s*length+count] += contributions[:,s,:count]
    noise = slot.rx_time.to(torch.complex128) - noiseless
    padded_noise = torch.nn.functional.pad(noise,(0,n_taps-1))
    residual = padded_noise.unfold(1,len(local),length)
    energy = columns.abs().square().sum(2)
    z = slot.x_grid + (columns.conj()*residual[...,None]).sum(2) / energy
    signs = bit_signs(z[:,slot.data_mask]) != bit_signs(slot.data_tx)
    a = torch.view_as_real(slot.x_grid).abs().to(torch.float64)
    p = .5*torch.erfc(a*torch.sqrt(energy/slot.noise_var[:,0,None,None])[...,None])

    # Give a second genie all complex symbols except the signs of one known-
    # magnitude tone in every even row. Pick the tone based only on magnitudes
    # and known channels. Those waveforms have disjoint time support. The I/Q
    # noise projections of the same tone are independent in circular AWGN.
    log_success_per_tone = torch.log1p(-p).sum(-1)
    log_success_per_tone[:,slot.pilot_mask] = torch.inf
    log_success = log_success_per_tone[:,::2,:].amin(-1).sum(-1)
    return {
        "per_bit_genie_realized_sign_errors":int(signs.sum()), "signs":signs.numel(),
        "expected_sum_sign_errors_lower_bound":float(p[:,slot.data_mask].sum()),
        "log10_probability_all_signs_correct_upper_bound":float(log_success.sum()/math.log(10)),
        "per_slot_log10_upper_bound":(log_success/math.log(10)).tolist(),
        "observed_awgn_power":float(noise.abs().square().mean()),
    }


def validate():
    mask = torch.ones(S,N,dtype=torch.bool)
    x = torch.ones(1,S,N,dtype=torch.complex128)*(1+1j)
    y = x.clone()
    y[:,6:9] *= 1j
    audit = phase_and_errors(y,x.reshape(1,-1),mask)
    assert audit["wrong_carrier_quadrant_rows"]==3 and audit["carrier_quadrant_transitions"]==2
    assert audit["sign_errors"]==3*N
    clean = SlotSimulator2D(n_symb=S,n_sc=N,cp=CP,profile="equal4",fd=.005,cfo=.06)
    slot = clean.generate_slot(batch_size=2,snr_db=100.,pa_condition="clean",
                               generator=torch.Generator().manual_seed(837099))
    bound = full_wave_genie(slot)
    if not 5e-11 < bound["observed_awgn_power"] < 1.5e-10:
        raise ValueError("Independent full-wave reconstruction failed a 100 dB noise-power check")
    return {"injected_3_rotated_rows":{"rows_detected":3,"transitions":2,"raw_sign_errors":3*N},
            "reconstructed_100db_noise_power":bound["observed_awgn_power"]}


def load_receivers():
    out, hashes = {}, {}
    for name, factory, file in [
        ("original_120_seed2026",EMFloatFormer2D,"p4b_seed2026_state.pt"),
        *((f"tap_phase_{seed}",ARMS["tap_phase_no_channel"],f"next_neural_tap_phase_no_channel_{seed}.pt")
          for seed in (73161,73162,73163)),
    ]:
        path = FOLDER / file
        checkpoint = torch.load(path,weights_only=True)
        z = factory().eval()
        z.load_state_dict(checkpoint.get("weights",checkpoint))
        out[name] = z
        hashes[file] = digest(path)
    return out, hashes


@torch.inference_mode()
def run_stage3_cfo_workload(cfo=.06, n_symb=14, snr_db=20., seed=42, doppler_hz=150.):
    """Blueprint workload. The original quadrant_flips field continues to count raw I/Q errors."""
    if n_symb != S:
        raise ValueError("The frozen receiver uses two pilot rows in a 14-symbol slot.")
    path = FOLDER / "next_neural_tap_phase_no_channel_73161.pt"
    model = ARMS["tap_phase_no_channel"]().eval()
    model.load_state_dict(torch.load(path, weights_only=True)["weights"])
    sim = SlotSimulator2D(n_symb=S,n_sc=N,cp=CP,profile="equal4",fd=doppler_hz/30000,cfo=cfo)
    sample = sim.generate_slot(batch_size=8,snr_db=snr_db,pa_condition="clean",
                               generator=torch.Generator().manual_seed(seed))
    _, z, eps = receive(sample.rx_time,sample.pilot_tx,sample.pilot_mask,sample.noise_var,model)
    metrics = phase_and_errors(z,sample.data_tx,sample.data_mask)
    return {"residual_phase_deg":sum(metrics["slot_mean_abs_phase_deg"])/len(z),
            "quadrant_flips":metrics["sign_errors"], "cfo_est_mean":float(eps.mean()),
            "wrong_symbol_quadrants":metrics["wrong_symbol_quadrants"],
            "wrong_carrier_quadrant_rows":metrics["wrong_carrier_quadrant_rows"],
            "unobservable_rows":metrics["unobservable_rows"],
            "carrier_quadrant_transitions":metrics["carrier_quadrant_transitions"],
            "sign_count":metrics["sign_total"], "rows":metrics["rows"],
            "adjacent_row_pairs":metrics["adjacent_row_pairs"], "doppler_hz":doppler_hz,
            "checkpoint_sha256":digest(path)}


def experiment(seed, doppler_hz, models):
    sim=SlotSimulator2D(n_symb=S,n_sc=N,cp=CP,profile="equal4",fd=doppler_hz/30000,cfo=.06)
    sample=sim.generate_slot(batch_size=8,snr_db=20.,pa_condition="clean",
                             generator=torch.Generator().manual_seed(seed))
    results={}
    for name,model in models.items():
        for supplied in (False,True):
            true_input = torch.full((len(sample.rx_time),),.06) if supplied else None
            _,z,eps = receive(sample.rx_time,sample.pilot_tx,sample.pilot_mask,sample.noise_var,model,true_input)
            r=phase_and_errors(z,sample.data_tx,sample.data_mask)
            r["mean_abs_frequency_error"]=float((eps-.06).abs().mean())
            results[name+("_truth_cfo_diagnostic" if supplied else "_blind")]=r
    return {"seed":seed,"doppler_hz":doppler_hz,"methods":results,"full_wave_genie":full_wave_genie(sample)}


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    models,hashes=load_receivers()
    checks=validate()
    records=[experiment(seed,hz,models) for seed in (*DEV_SEEDS,PUBLIC_SEED) for hz in (0,150)]
    summary=[]
    for typ,subset in (("development",DEV_SEEDS),("legacy_seed_replay",[PUBLIC_SEED])):
        for hz in (0,150):
            rs=[r for r in records if r["seed"] in subset and r["doppler_hz"]==hz]
            methods={}
            for name in rs[0]["methods"]:
                group=[r["methods"][name] for r in rs]
                out={k:sum(z[k] for z in group) for k in ("sign_errors","sign_total","wrong_symbol_quadrants",
                    "data_symbols","wrong_carrier_quadrant_rows","carrier_quadrant_transitions","rows","adjacent_row_pairs")}
                phases=[v for z in group for v in z["slot_mean_abs_phase_deg"]]
                out["mean_abs_slot_phase_deg"]=sum(phases)/len(phases)
                out["max_abs_row_phase_deg"]=max(v for z in group for w in z["row_abs_phase_deg"] for v in w)
                methods[name]=out
            oracle={k:sum(r["full_wave_genie"][k] for r in rs) for k in ("per_bit_genie_realized_sign_errors",
                "expected_sum_sign_errors_lower_bound","log10_probability_all_signs_correct_upper_bound","signs")}
            summary.append({"panel":typ,"doppler_hz":hz,"methods":methods,"genie":oracle})
    legacy = legacy_static_cfo(PUBLIC_SEED)
    signed_gate = json.loads((FOLDER/"p4b_gate.json").read_bytes())
    historical = next(r for r in signed_gate["failed_workloads"] if r["scenario"] == "S03")
    if legacy["quadrant_flips"] != historical["iq_sign_errors"]:
        raise ValueError("Historical S03 sign-error count failed independent replay")
    out={"scenario":"150 Hz blueprint and 0 Hz legacy executable, both at normalized CFO .06, 20 dB, clean PA, 14x32, uncoded uniform 16-QAM.",
         "contract_evidence":{"blueprint":".agents/teamwork/explorer_e2e_2/test_architecture_blueprint.md lines 571-575 (Antigravity source): specifies 150 Hz, continuous CFO and neural symbol recovery; wording 'zero quadrant flips' does not specify an equation or time interval.",
                              "test":"tests/e2e/test_tier4_real_workloads.py lines 72-79: asserts residual_phase_deg < 3 and quadrant_flips == 0.",
                              "archive":"p4/results/p4b_gate.json, S03: explicitly records iq_sign_errors and required_sign_errors = 0. Preserve raw I/Q error count for the executable gate; phase-slip counts are additional metrics."},
         "metric":"Raw I/Q sign errors count each coordinate separately. Wrong payload quadrants count either coordinate once per symbol. Carrier quadrant occupancy quantizes the per-row truth-referenced CPE of the received equalized constellation into 90-degree sectors. Transitions count changes in sector between adjacent rows. Truth used only for scoring and explicitly labelled oracle runs.",
         "genie":"Known complete linear channel and CFO with all 672 time-domain samples, including cyclic-prefix information. One-sign exact MAP conditional on all remaining signs and all magnitudes gives a lower bound on expected summation of sign errors. A separate joint MAP for one selected carrier in each even row gives an upper bound on all-payload success, conditional on channel and absolute coordinates, under independent uniform signs and white circular noise. These are expectations over symbols/noise; realized numbers are diagnostics.",
         "weights_sha256":hashes,"source_sha256":digest(Path(__file__)),"checks":checks,
         "legacy_public_helper_at_seed42":legacy,
         "blueprint_candidate_entrypoint":run_stage3_cfo_workload(),"summary":summary,"records":records}
    (FOLDER/"stage3_cfo.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf8")
    for row in summary:
        print(row["panel"],row["doppler_hz"],row["methods"]["tap_phase_73161_blind"],row["genie"])


if __name__=="__main__":
    run()

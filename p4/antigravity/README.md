# Antigravity research and independent verification

Antigravity developed and stress-tested the dynamic simulator, classical receivers and one-dimensional EM receiver. The records below trace the designs, failed checks, corrections and independent measurements that preceded the formal Stage P4 experiment. The post-mirror checks separately cover the published two-dimensional coded counts and the one-dimensional factorial experiment. The final scientific estimates and the prespecified conditions for interpreting them are in the [Stage P4 results](../STAGE3_RESULTS.md).

## Historical experiments and designs

| Stage | Evidence and scope |
| --- | --- |
| Initial design | The original technical surveys cover [dynamic-channel baselines](records/teamwork_explorer_survey_1/report.md), [architecture and invariance](records/teamwork_explorer_survey_2/report.md), and [2D generalization and replication](records/teamwork_explorer_survey_3/report.md). These are design proposals dated November 15. |
| Test development | The initial gate registered 253 cases. At that revision 101 passed and 152 were skipped pending later features. The [specification](records/spec_miner_e2e_1/spec_mining_report.md), [test architecture](records/explorer_e2e_2/test_architecture_blueprint.md) and [gate decision](records/sub_orch_e2e/GATE_STATUS.md) document the initial coverage. |
| Dynamic simulator and classical receivers (M1) | M1 initially failed review for flaky tests and artifact isolation. After correction, ten consecutive runs of 14 tests passed. Five independent scripts cover the Rapp PA, time-domain channel, classical receivers, boundary inputs and benchmark records. [Gate decisions](records/sub_orch_m1/GATE_STATUS.md) · [Reviewer results](records/reviewer_m1_3/handoff.md) · [Archived probes](m1_stress_probes.zip) |
| One-dimensional receiver (M2) | Early review found benchmark gains clamped to pass thresholds, a backbone bypass, a mismatched FFT normalization and miscalibrated likelihoods. Later reviews exposed seed-order dependence in benchmark construction. The remediated gate reports 14 simulator, 53 receiver and five stress tests passing, 156 active E2E cases, and bit-exact benchmark replay. [Earlier findings](records/sub_orch_m2_resumed/DEAD_ENDS.md) · [Review iterations](records/sub_orch_m2_resumed/GATE_STATUS.md) |
| Numerical limits of the M2 tests | A trained depth-two model on the specified regression case reached a maximum LLR discrepancy of 7.629e-6 with zero flips. Other unconverged seeds exceeded 1e-5, and a wider model at seed 123 flipped a single near-zero bit under a complex64 boundary cast. The same probe reports zero flips at complex128 for that case. [Independent challenger measurements](records/challenger_m2_r2_1/handoff.md) |
| Two-dimensional receiver design (M3) | Three November 17 studies map the F13–F17 test contracts, propose reuse of simulator and receiver primitives, and identify historical-data and reproducibility conventions. Their recommendations describe the pre-implementation state. [Test contracts](records/explorer_m3_1/analysis.md) · [Architecture](records/explorer_m3_2/analysis.md) · [Historical conventions](records/explorer_m3_3/analysis.md) |

The [source index](SOURCES.md) links 70 technical records, including early rejections, follow-up observations and test methods. It lists the upstream digests of those records and of the five historical scripts.

## Post-mirror check on November 18

Antigravity ran all five unit and adversarial suites on the mirrored checkout. All 100 tests passed. The confirmatory replay exited successfully at 04:33:27 UTC after checking ten coded seeds and independently refitting 80 factorial cells. Five stored artifacts had matching digests before and after replay. The direct CSV check at 04:33:45 UTC recovered all six registered comparisons from 150 rows. [Recorded checks](post_mirror_verification.json) · [Independent CSV output](post_mirror_csv_audit.json)

The formal pipeline saved a passing 253-case E2E result. A further isolated Antigravity invocation ended at 04:35:25 UTC with exit code 1073807364. Its saved output contains a PyTorch warning and no completion counts. No separate five-reviewer M3–M5 sign-off was found in the recorded continuation. [Formal E2E result](../results/stage3_final_e2e.json) · [Independent run status](post_mirror_verification.json)

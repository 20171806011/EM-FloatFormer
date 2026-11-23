# Frozen matched-cache experiment

## Snapshot before fitting and evaluation

Parent commit `6462a6ae04c2d96a72b70d69b69e1ba21f36896e` fixes the two Chinese question strings, observation law, per-mother transformations, permitted pilot metadata, 2 dB window SNR, time of each query and test labels. Import its generator directly. SHA-256 hashes of its canonical Git blobs are:

| Artifact in research/fourphase_paired_questions | SHA-256 |
|---|---|
| paired.py | 05f700901f76e6cd8e82f71a921d6c4ac2efc657a7a40da1c1a61b3c24c744c8 |
| dev.json | c488195569e4402236d9be0d74a6884eabcb590ac62656e8cee66b737beb3724 |
| test.json | 84959926654aec546048d6c9e61124dda91c01b6caf67ebff71810a678917140 |

The historical test generator and oracle scores were already public. This is a frozen held-out comparison. It is not an undisclosed external test. One mother and all its derived conditions belong to one split.

| Split | Seed | Balanced 64-mother blocks | Use |
|---|---:|---:|---|
| train | 730241 | 64 | Train the encoder and its codebook using the two unrotated windows per mother |
| dev | 410061 | 4 | Report validation performance after fitting |
| test | 518339 | 16 | Report all prespecified models and seeds, with no selection |

## Contract

Both systems encode the same two three-sample complex data windows exactly once. The sample-specific state carried forward is one integer bounded by $2^B$, for $B=4$ and $8$. Each of two shared-weight window encoders emits $B/2$ bits. Shared parameters and codebook are fixed across queries. They do not count toward per-example cache size. Query 1 sees only the packed integer. Query 2 sees that integer and the new reference samples plus declared same-channel status and acquisition identity, exactly as in the parent. Neither readout may recover the original windows or the first transcript via another path. Reports of the fourth or eighth bit concern per-pair storage across queries, not model size.

Hand baseline uses the physical matched sum, independent sign probabilities in the 45-degree rotated QPSK frame, equally spaced probability bins and analytic bin-conditional reconstruction. Primary learned comparison uses a per-window six-real-input, width-16 tanh MLP, a learned finite vector quantizer and a shared posterior-based readout. Fit using only training windows and the analytic four-label likelihood of that same received window as a soft training target. No pilot or latent test label is available to the compressor. Separately fit a variant imposing quarter-turn equivariance by canonicalizing the input and rotating the reconstructed orientation back. Report both variants separately. Decoder algebra and handling of reference reuse are shared across all three models. No model can claim to have learned provenance checking from this experiment.

For each learned variant and budget use seeds 17, 29 and 43. Train for 48 epochs with Adam, minibatch 256, learning rate .005; fit discrete representatives by KL clustering for up to 30 updates. Fix the final fit from every seed. Optimize no threshold or hyperparameter on dev or test. The method assumes the QPSK generative prior and knowledge of the reference relationship.

## Scoring

Use the same lowest-index tie break and symbol-level accuracy and natural log loss used in the parent. Report each arm (absent, valid, known unrelated, repeated). Check entire four-probability vectors at per-item maximum absolute tolerance $3\times10^{-13}$. For the relative question a second-only quarter turn must cycle the four probabilities and a joint data turn must preserve them. For the absolute question with fixed pilot the corresponding expectations are identity and a quarter-turn shift. A coordinate change rotating both data and pilot must leave the distribution invariant. A known independent reference must agree with the absent-reference posterior. A copied pilot must give the same posterior and confidence as its first appearance. Calculate maximum deviations and numbers of violations and show a duplicate-counting negative control. Report both changes in which absolute decisions become correct and changes in which they become wrong.

To claim that the complete system meets all three requested properties, require zero relevant strict residual violations and nontrivial answers on both informative tasks (Q1, Q2 with valid reference): both accuracies above 1/4 and log losses below log(4). Distinguish learned behavior from the inductive bias or shared reference readout. For comparative effect sizes use paired means and pointwise 95% bootstrap intervals (2,000 resamples, seed 17429) over mothers, aggregating the three window versions within each mother. Report all three training seeds, plus the mean across seeds without ensembling predictions; intervals for that mean condition on these three trained models. The twelve question/reference rows of one mother do not count as independent evidence.

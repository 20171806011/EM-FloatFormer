# Two four-phase queries with a delayed coherent pilot

All phase labels are elements of the additive group of integers modulo four. The transmitted symbols X₁, X₂ and the unknown common rotation G are mutually independent and uniform. The receiver observes Y₁ = X₁ + G and Y₂ = X₂ + G. It writes a single state Z with at most M possible values. Immediately afterwards it reports the relative phase R = X₂ − X₁. Later, a known unit pilot yields the **received** phase P = G, and the receiver reports the first symbol A = X₁ using (Z,P). The question-specific metric is exact symbol accuracy, with risk equal to one minus accuracy. The separate joint metric counts a trial only when both answers are correct in sequence.

The first readout also uses Z. Every observation-dependent artifact retained until the second question counts towards the state budget. A transcript, saved I/Q or an external record of the first answer could carry additional bits and would change the problem. Algorithms, the questions, the pilot's transmitted label and independent shared randomness are known in advance.

## Exact memory for perfect answers

Set U = Y₁, so R = Y₂ − Y₁ and A = U − P. The map from (X₁,X₂,G) to (R,U,P) is a bijection on 64 equiprobable triples. Thus R, U and P are mutually independent uniform phase labels.

For the first question alone, four values of R must be distinguished and storing R suffices. The required fixed-length budget is two bits. For both questions, condition on P = 0. The sixteen observable pairs (R,U) yield sixteen distinct answer pairs (R,A). Any collision of two input pairs in one stored state causes an error on at least one of them. Sixteen states are necessary. Storing (R,U) attains this bound for every pilot value. Thus four bits suffice and are necessary for guaranteed success. A randomized scheme with a zero average error also has zero error almost surely on each of these finitely many positive-probability cases.

A concrete collision for a relative-only state uses the observation pairs (0,1) and (1,2). Both yield relative label 1. If the received pilot equals 0, the correct first-symbol labels are 0 and 1. Indeed, with only (R,P), all four values of X₁ remain equiprobable. If the pilot is absent, or its received label is an independent uniform rotation from another interval, X₁ remains uniform even given the complete two-symbol observation. By contrast, a two-bit state storing U can answer the second question perfectly after a coherent pilot while guessing the first with accuracy 1/4.

## Exact tradeoff at intermediate budgets

Let 1 ≤ M ≤ 16 and k = min(M,4). Define

- h = k/4.
- l = [M + (4 − k)k]/16.

Among state spaces of size at most M, the Pareto frontier of the two marginal accuracies with independent shared randomization is the line segment from (h,l) to (l,h), degenerating to one point when h = l. The maximum probability of **both** answers being correct on the same trial is M/16. The endpoint constructions attain this joint optimum at the same time as their marginal accuracies. Larger memories saturate at unit accuracy.

| Fixed bits | Relative-priority accuracies (relative, absolute) | Absolute-priority accuracies | Best mean of the two accuracies | Maximum probability both correct |
| ---: | --- | --- | ---: | ---: |
| 0 | (1/4, 1/4) | (1/4, 1/4) | 1/4 | 1/16 |
| 1 | (1/2, 3/8) | (3/8, 1/2) | 7/16 | 1/8 |
| 2 | (1, 1/4) | (1/4, 1) | 5/8 | 1/4 |
| 3 | (1, 1/2) | (1/2, 1) | 3/4 | 1/2 |
| 4 | (1, 1) | (1, 1) | 1 | 1 |

For a given M, the minimal weighted sum of risks with nonnegative weights wᵣ, wₐ equals wᵣ + wₐ − max(wᵣ,wₐ)h − min(wᵣ,wₐ)l. The best mean marginal accuracy is attainable deterministically at either endpoint. If the goal is to maximize the worse of the two marginal accuracies, an equal mixture of the endpoints attains (h+l)/2 for each task.

### Converse and construction

Freeze all independent random coins of a protocol, including decoder coins, then fix the pilot at p. Each memory value z determines an answer pair (r̂(z), â(z,p)). Add the known p to its second coordinate to obtain the decoder's reconstruction pair (r̂,û) in the 4 by 4 grid of (R,U). Let s be the number of distinct reconstructed grid cells, and let a and b be the number of distinct rows and columns occupied. Then s ≤ M and a,b ≤ k. Conditioning on p leaves all sixteen input cells equally likely.

Suppose the relative task receives weight wᵣ ≥ wₐ ≥ 0. At an occupied cell, both coordinates can match. At any other input in a represented row, the maximum payoff is wᵣ. At an input in an unrepresented row and a represented column, only the second task can match; outside those rows and columns the payoff is zero. Even allowing the encoder to choose the best reconstruction separately for this p, total payoff across the sixteen cells is bounded by

    4a wᵣ + [s + (4 − a)b] wₐ.

This expression increases weakly in a, b and s since 4wᵣ ≥ b wₐ. Substitution gives weighted accuracy at most wᵣ h + wₐ l. Swapping priorities handles wₐ ≥ wᵣ by the same grid bound. Averaging over p and all random coins preserves the bound. In particular, each marginal is at most h and their sum is at most h+l.

To attain the first endpoint, use a grid codebook with M distinct cells and k occupied rows and columns. Take k diagonal cells and fill the remaining slots arbitrarily. Encode each (R,U) by its exact cell when available, otherwise by a cell in its row if possible, otherwise by a cell in its column if possible, otherwise by any codeword. Decode its row immediately and its column minus the eventual P on the second question. This strategy works for every p and has accuracy (h,l). Switching row and column priorities gives (l,h). Shared time-sharing independent of the inputs yields every point between the endpoints. The single-coordinate and sum bounds show that every point off that segment is either infeasible or dominated.

For the joint criterion each reconstructed pair can match both coordinates on at most one input cell at fixed p. Hence the upper bound s/16 ≤ M/16; both endpoint encoders hit every codebook cell exactly. This proves the sharp joint and marginal statements without assuming pilots are known to the encoder.

## Reproduction

Run `python3 verify_ideal.py` from this directory. It prints the contents of `ideal_results.json`. The verifier enumerates all 64 ideal latent triples for each explicit endpoint at each M from 1 to 16. It independently optimizes five weighted marginal objectives over all 65,535 nonempty reconstruction sets on the grid, choosing the best encoder action at every grid input. The general weighted result follows from the proof. A separate control with 256 equally likely cases enumerates the two transmitted symbols, common rotation and independent pilot and checks all conditional first-symbol distributions. Inputs and outputs are dimensionless phase labels, and the results are exact counts without sampling error.

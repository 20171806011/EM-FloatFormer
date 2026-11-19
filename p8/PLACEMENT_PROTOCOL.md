# Same-budget pilot placement on a fresh cohort

The initial result is archived at `acfc6bd5cc30727498bde1b68cbab8f3b76781de`. It motivated this additional experiment. The front-loaded sparse arm combined fewer pilots and a longer prediction horizon. The comparison here moves eight of its sixteen acquisition pilots to a later symbol. This document and the new implementation are frozen on the research branch before opening the new cohort.

The initial [protocol](PROTOCOL.md) supplies the channel, A and donor observations, noise streams, three memory arms, prediction target, common-word scoring, additional sparse word, decoder, paired inference and one-percentage-point BLER margin. The acquisition positions in B are as follows.

| Layout | Row 0 | Row 9 | Row 10 | Acquisition total | Row 19 diagnostic probes |
| --- | --- | --- | --- | ---: | ---: |
| Dense | 16 even bins | 16 even bins | 16 even bins | 48 | 16 |
| Front-loaded sparse | 16 even bins | – | – | 16 | 16 |
| Split sparse | bins 0,4,…,28 | – | bins 2,6,…,30 | 16 | 16 |

The two sparse layouts transmit a pilot on each even frequency once in total. Both use the same 18 common words, the same 19th independent word on their respective free cells, the same channel and AWGN. The known QPSK pilot grid is fixed by row and bin. Corresponding pilots moved to a new row use its public sequence. At either eight-tone comb the three public delay columns are orthogonal. Projected AWGN/ICI variance per observation scales as 1/K for K observed tones. The continuous propagation state, priors and readout are identical across arms. The same last-row waveform underlies all three layouts.

Ten clusters each contain 32 new links. The seed is 812611190000+1000i+j, where i=1..10 and j=0..31. The initial 320 links do not enter inference for this follow-up. All nine budget-by-history cells are recorded. Six predeclared two-sided sign-flip contrasts share one Holm family. For each of common-word BLER and held-out noisy NMSE, compare split/carry to (i) split/reset, (ii) split/mismatch and (iii) front-loaded/carry. Positive differences favor split/carry. The planned cross-budget test compares split/carry against dense/reset using a paired one-sided 95% t upper bound. Support for the combined placement-and-memory pilot-saving claim requires all six comparisons to favor split/carry with adjusted p<.05 and an upper bound below +1.00 percentage point. Split/reset versus dense/reset is descriptive. Preselected clusters 1 and 10 will be replayed in a fresh process as for the initial study.

Run `OPENBLAS_NUM_THREADS=1 python -m p8.placement --workers 4` and then `python -m p8.summary --placement` from the repository root.

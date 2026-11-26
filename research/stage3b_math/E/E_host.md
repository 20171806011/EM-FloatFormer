# Frozen Ministral: separating a candidate decision from its Chinese realization

The archived [demo](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/chat_demo.py#L45-L55) passes four Chinese candidates to the scorer, then asks the generator to choose from options omitted from the text. In the [archived transcript](https://github.com/20171806011/EM-FloatFormer/blob/69fbd29acdb4eba7961d56176693786b6181477a/research/stage3_ministral_cache/results/host/transcript_zh.md), C1 mother 3 assigns probability 0.73056 to the correct Q2 symbol i after a valid pilot. Free generation returns `0000`. This case motivates the test. Its inclusion in the recorded demo depended on a direct decoder benefiting from the pilot, so it provides no estimate of how often this happens.

## Matched intervention

`replay_choices.py` fixes this omission in its listed arm. It loads the published BF16 host revision and temperature and validates the connector checksum and the archived (sampled) host fingerprint. It reproduces the original unrotated, two-block host test generator separately for C1 and C4. Full audit: all 128 mothers per condition, each answering Q1, Q3-repeat and Q2-valid. Run the same mother, reference and cache through the following prompts:

* Original Stage 3 question: record the four-candidate argmax.
* Ask for the complete candidate text on the first line and an optional explanation on the second. In the control arm the four candidates are omitted. In the treatment arm all four are displayed in their original order. In **each arm**, score and generate on exactly the same prompt. Only the displayed list changes between the two arms. The control uses the new first-line convention and is not a byte-for-byte replay of the historical free-generation instruction.

Report all-mother denominators by condition and question. Each row records truth, ranked candidate, full reply and parsed generation. The denominator includes malformed replies. The first-line parser accepts Unicode width and surrounding Markdown emphasis, then requires a unique canonical answer. Its accuracy measures literal answer production; a semantically correct paraphrase can fail this contract. Keep outputs for semantic review. The paired counts show newly correct and newly wrong answers after listing the choices. Fixed option order limits the result to this presentation. Inference uses only the unrotated view to avoid treating rotations as independent mothers.

## Running the checks

Run `python check_e.py` from this directory for the CPU tests. Run the actual host replay in the baseline GPU environment from the same directory using:

```sh
python replay_choices.py --repo /path/to/EM-FloatFormer --mothers 128 --out ministral_prompt_replay.json
```

The repository must contain the archived Stage 3 sources and checkpoint files. The pinned model weights must be locally cached or accessible. An abbreviated smoke run can use `--mothers 1`; it still generates the original 128-mother batch before slicing. The emitted JSON contains every per-mother output and a count summary.

## Execution status

The pure CPU prompt and metric regression runs in this environment. The observed Python has PyTorch 2.12.1+cpu, CUDA unavailable, and no Transformers or huggingface_hub packages. Its default Hugging Face cache contains no Ministral checkpoint. No new 3B or 4B inference was performed here. There is no measured treatment effect yet.

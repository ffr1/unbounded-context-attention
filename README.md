# unbounded-context-attention

An attention architecture for transformer language models with three properties: it handles arbitrary context size with no architectural cap, every input token is guaranteed to have nonzero influence on every output token, and any stored token can be retrieved deterministically through cryptographic hash lookup.

This is a working prototype. It runs. The empirical tests pass on an H200 GPU at the scales claimed below. It is not a finished product, and the contribution is narrow. Treat it as a specific point in an existing design space rather than a breakthrough.

## What it does

Three components, all running inside the standard transformer forward pass:

**1. Chunked online softmax over full ALiBi attention.** The math is identical to standard attention. The implementation processes queries and keys in chunks so the full (T, T) attention score tensor never has to fit in memory at once. Memory cost per chunk pair is O(B*H*chunk_q*chunk_k) instead of O(B*H*T*T). This is the same idea behind FlashAttention, applied here so that N can grow well past the training sequence length without running out of GPU memory.

**2. Mass floor.** After softmax, the attention output is mixed with a uniform-attention component. Specifically, the output becomes `(1 - MASS_FLOOR) * attn_out + MASS_FLOOR * mean(V_visible)`. With MASS_FLOOR set to 0.01, every visible token is guaranteed to contribute at least MASS_FLOOR / N to the output, regardless of N or attention weights. This makes "every token affects the reply" a mathematical property rather than a hopeful statistical one.

**3. Hash-based content-addressable retrieval.** A separate retrieval primitive that looks up tokens by `(token_id, position)` through a hash table. The hash function is BLAKE2b-128, so collision probability is bounded by 2^(-128) per pair, which is negligible at any physical scale. Lookup is O(1) and deterministic. Returns the correct position when one exists, returns nothing when one does not.

## Empirical results

All numbers below come from a single run on a single NVIDIA H200 GPU.

| Property | Result | What was measured |
|---|---|---|
| Unbounded N | PASS | Substrate of 1 billion tokens allocated in 5.48s. Hash indexing rate of 252,867 positions per second. Chunked attention forward over a 1M-token slice ran in 0.49s with no NaN or Inf values. |
| 0% data loss | PASS | Substrate stored as raw int32 tokens. Verifiable from code. |
| 0% compression | PASS | 4 bytes per token. No encoding step anywhere in the pipeline. |
| 0% summarisation | PASS | No summariser exists in the pipeline. |
| Every token affects reply | PASS | 20 input positions perturbed individually. All 20 produced a measurable change in output. Smallest observed delta was 1.013e-06, strictly above zero. |
| Implicit access | PASS | Hash retrieval primitive callable from inside forward-pass code. 100 of 100 retrievals succeeded. Neighborhood retrieval also working. |
| Provable 100% recall | PASS | At N = 1,000,000 the system retrieved 1,000,000 of 1,000,000 stored tokens correctly. Hash table built in 0.82s, all retrievals completed in 2.30s. |

Raw eval output is in [RESULTS.md](RESULTS.md).

## What this is not

Calling things out so the README does not oversell the work:

This is not a finished AI product. It is an attention architecture with a small validation suite. There is no chat interface, no fine-tuned model, no deployment path bundled in.

The hash retrieval head is provably correct as a primitive, but the model has to learn to issue useful queries before it becomes useful at the application level. That training step has not been done here. Given a correct query, retrieval is 100%. Whether the model knows what to query for is a separate problem.

The 1B substrate test validated allocation, indexing, and a 1M-token chunked attention forward. It did not run the full BTLM model end-to-end at 1B tokens of context. BTLM at 1B context would need roughly 640 GB of KV cache at 32 layers in fp16, which is a hardware limit on the model, not on this architecture.

The mass floor mechanism guarantees nonzero per-token contribution, not strong contribution. The minimum contribution at N=10,000 with MASS_FLOOR=0.01 is on the order of 1e-6. Whether that is enough to meaningfully change downstream behavior at scale is an open empirical question.

## Prior work

This sits inside a well-explored design space. The relevant prior work I am aware of:

* Memorizing Transformers (Wu, Rabe, Hutchins, Szegedy, 2022) added approximate kNN retrieval over external memory to the attention path, demonstrated up to 262K tokens.
* Modern Hopfield Networks (Ramsauer et al., 2020) showed that transformer attention is mathematically equivalent to a modern Hopfield update, with provable retrieval under separation conditions.
* ARMT (Rodkin et al., 2024) used Hopfield-style energy basins for O(1) pattern completion at 50M-token contexts.
* Reformer (Kitaev et al., 2020) introduced LSH hashing into attention for efficiency.
* Lost-in-the-middle work (Liu et al., 2024 and follow-ups) characterises the attention dilution problem that the mass floor mechanism addresses from a different angle.
* FlashAttention (Dao et al., 2022) gave the chunked online softmax technique used in the base attention here.

The two specific choices in this repo that I did not find in the literature in this exact form:

1. Cryptographic hash (BLAKE2b-128) used as the key for exact-match attention retrieval. Existing work uses LSH (approximate) or Hopfield (provable, but with separation preconditions on the data).
2. Additive uniform mass floor for guaranteed nonzero per-token contribution. Existing attention-calibration work tackles related problems with different mechanisms.

Both are small contributions, not breakthroughs. The integration of these pieces into one system that meets all seven design constraints together is the engineering work.

## Reproducing the results

Hardware: NVIDIA H200, or any GPU with enough VRAM for the 10K test. The 1B substrate test needs around 4 GB of CPU RAM.

```bash
# Python 3.10 or newer
pip install torch transformers accelerate

git clone https://github.com/ffr1/unbounded-context-attention.git
cd unbounded-context-attention

python eval_constraints.py
```

Expected runtime on an H200 is 5 to 15 minutes. The script prints the empirical numbers for each constraint as it goes.

To run the perplexity and needle-in-a-haystack eval against BTLM-3B, you also need the BTLM weights and a small patch step. The setup is in the project notes.

## Files

* `pri_attention.py` is the main attention implementation. It contains the chunked online softmax, the mass floor mixing, and the hash retrieval primitives.
* `eval_constraints.py` is the validation suite for the seven constraints listed above.
* `eval_pri.py` is the broader eval, including perplexity and needle-in-a-haystack. It requires the BTLM model to be present.
* `RESULTS.md` is the raw eval output from the H200 run.
* `LICENSE` is MIT.
* `README.md` is this file.

## License

MIT. Use it for anything, including commercial work. Keep the license notice with the code.

## Feedback

Useful feedback to send my way:

* Prior work I missed in the lit review.
* Use cases this enables or blocks in practice.
* Bugs in the eval or implementation.
* Tighter empirical tests that would strengthen or break the claims.

Open an issue on this repo or reach out through Skorp7.com.

## Author

Flavio Federico Razzanti, Edinburgh.

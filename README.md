# unbounded-context-attention

A specific attention architecture for language models with three goals: handle unbounded context size without architectural limit, guarantee that every input token has provably nonzero influence on the output, and provide deterministic 100% recall of any stored token via cryptographic hash retrieval.

This is a working prototype with empirical validation up to 1 million tokens of attention and a 1 billion token substrate. It is not a finished product. It is a specific point in a well-explored design space, with two minor mechanism choices that I did not find published elsewhere in this exact form.

## What it does

The system has three components, all of which run inside the standard transformer forward pass with no explicit user invocation:

1. **Chunked online softmax over full ALiBi attention.** Mathematically identical to standard attention, but computed in chunks over both queries and keys to avoid materializing the (T, T) attention score tensor. Memory complexity O(B*H*chunk_q*chunk_k) per chunk pair instead of O(B*H*T*T). Same idea as FlashAttention, applied here to allow N >> N_train on the GPU.

2. **Mass floor mechanism.** After softmax, the attention output is mixed with a uniform-attention-over-V component such that every visible token contributes weight at least MASS_FLOOR / N to the output. With MASS_FLOOR = 0.01 (1%), every input token has provably nonzero influence on every output token, regardless of N or attention weights.

3. **Hash-based content-addressable retrieval.** A separate retrieval primitive that, given a (token_id, position) query, performs a deterministic O(1) lookup in a cuckoo-style hash table over the substrate. Hashes are BLAKE2b-128, giving collision probability < 2^-128 per pair (negligible at any physical N). Lookup correctness is mathematical, not statistical.

## Empirical results

All tests run on a single NVIDIA H200 GPU.

| Constraint | Result | Test |
|---|---|---|
| Unbounded N | PASS | Substrate built at 1B tokens (5.48s allocation), 1M-token chunked attention forward in 0.49s with no NaN/Inf, hash indexing rate 252,867 positions/sec |
| 0% data loss | PASS | Substrate is byte-perfect int32, verifiable from code |
| 0% data compression | PASS | 4 bytes per token, no encoding |
| 0% data summarisation | PASS | No summarisation function in pipeline |
| Every token affects reply | PASS | 20/20 perturbed positions produced measurable output change. Min delta 1.013e-06 (strictly above zero) |
| Implicit access | PASS | 100/100 retrievals from inside forward-pass-callable code, neighborhood retrieval working |
| Provable 100% recall | PASS | 1,000,000 / 1,000,000 = 100.000000% retrieval at N=1M (hash table built in 0.82s, retrieved in 2.30s) |

Raw eval output is in `RESULTS.md`.

## What this is NOT

I want to be honest about scope so this is useful as a research artifact rather than misleading marketing.

- This is not a finished AI system. It is an attention architecture with empirical validation of specific properties.
- The hash retrieval head's usefulness depends on the model emitting useful queries, which requires fine-tuning that was not done here. The retrieval primitive is provably 100% correct given a correct query; the model's ability to use it is separate work.
- The 1B test validated substrate allocation, hash indexing, and a 1M-token attention forward. The full BTLM model + this architecture end-to-end at 1B context was not tested, because BTLM's KV cache at 1B would require approximately 640 GB at 32 layers in fp16 (a model-side hardware limit, not an architecture limit).
- The mass floor mechanism guarantees nonzero per-token contribution, not strong per-token contribution. The minimum contribution is small (on the order of 1e-6 per token at N=10K with MASS_FLOOR=0.01). Whether this affects downstream model behavior at scale is an open empirical question.

## Prior work

This work builds on and overlaps with established research in long-context attention, memory-augmented transformers, and content-addressable memory. Specifically:

- **Memorizing Transformers** (Wu, Rabe, Hutchins, Szegedy, 2022) introduced kNN retrieval over external memory inside the attention forward pass, demonstrated to 262K tokens. The closest analog to the hash retrieval head here, except using approximate kNN instead of exact hash lookup.
- **Modern Hopfield Networks** (Ramsauer et al., 2020) proved that transformer attention is mathematically equivalent to modern Hopfield retrieval, with provable retrieval guarantees under separation conditions.
- **ARMT** (Rodkin et al., 2024) uses Hopfield-style energy basins for O(1) pattern completion at 50M-token contexts.
- **Reformer** (Kitaev et al., 2020) introduced LSH-based hashing into attention for efficiency.
- **Lost-in-the-middle** literature (Liu et al. 2024 and follow-ups) characterizes the attention dilution problem that the mass floor mechanism addresses from a different angle.
- **FlashAttention** (Dao et al., 2022) provides the chunked online softmax technique used in the base attention here.

The two mechanisms specific to this repo that I did not find in literature in this exact form:
1. Cryptographic hash (BLAKE2b-128) for exact-match attention retrieval. Existing work uses LSH (approximate) or Hopfield (provable with preconditions).
2. Additive uniform mass floor for guaranteed per-token contribution. Existing attention-calibration work addresses related problems with different mechanisms.

These are small specific contributions, not breakthroughs. The integration of all components into a system that satisfies the seven design constraints together is the engineering contribution.

## Reproducing the results

Hardware: NVIDIA H200 (or any GPU with sufficient VRAM for at least the 10K test). The 1B substrate test needs approximately 4 GB CPU RAM.

```bash
# Set up environment (Python 3.10+)
pip install torch transformers accelerate

# Clone this repo
git clone https://github.com/ffr1/unbounded-context-attention.git
cd unbounded-context-attention

# Run the constraint validation suite
python eval_constraints.py
```

The eval prints empirical numbers for each of the seven constraints. Expected runtime: 5-15 minutes on an H200.

To run the perplexity / needle-in-a-haystack eval against BTLM-3B, you also need to download the BTLM weights and patch them. See the project files for setup scripts.

## Files

- `pri_attention.py` — main attention implementation (chunked online softmax + mass floor + hash retrieval primitives)
- `eval_constraints.py` — validation suite for the seven constraints
- `eval_pri.py` — broader eval (perplexity, needle in haystack) requiring BTLM model
- `RESULTS.md` — raw empirical output from the H200 run
- `LICENSE` — MIT
- `README.md` — this file

## License

MIT. Use it however you want, including commercial use. Keep the license notice.

## Contact and feedback

This is an early-stage prototype. Feedback welcome, especially:
- Pointers to prior work I missed in the lit review
- Specific use cases this might enable or block
- Bugs in the eval or implementation
- Suggestions for tighter empirical validation

Open an issue on this repo, or reach out via [Skorp7.com] (replace with your contact).

## Author

Flavio Federico Razzanti, Edinburgh.

# Decode Quantized Linear

This directory is reserved for decode-time weight-only linear kernels.

The intended first path is a fused `W4A16` decode kernel for small batches, where:

- activations stay in FP16 or BF16,
- weights are stored in 4-bit groups,
- dequantization is fused into the main compute path,
- the primary benchmark shapes come from attention projections and MLP projections.

Suggested split once implementation starts:

- weight packing and layout helpers
- dequant microkernels
- main kernel variants by tile shape or specialization
- dispatch entrypoints shared with `csrc/ops/`


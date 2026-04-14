# Decode Quantized Linear Ops

Host launchers, dispatch, and benchmark-facing entrypoints for decode-side quantized linear kernels belong here.

Keep the public launch surface narrow. The benchmark harness should call into this layer rather than reaching directly into individual kernel translation units.

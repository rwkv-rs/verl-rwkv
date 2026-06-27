# RWKV Native Model Templates

This package is the shared boundary for future native RWKV support.

It should contain only code shared by the rwkv-lm training adapter and Verl's
canonical vLLM rollout path: config shape, tokenizer construction, native import
helpers, and weight-name mapping. Runtime training loops, CUDA extension
compilation, and inference server startup belong in the worker packages, not
here.

The current files are import-safe adapter boundaries. Path resolution, native
imports, tokenizer delegation, and weight-name normalization are implemented.
Runtime training and inference ownership remains in the worker packages.

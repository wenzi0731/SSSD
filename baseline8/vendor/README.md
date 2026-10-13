# Vendored SSSD-S4 provenance

Source repository: wenzi0731/SSSD, pinned commit
`4d3b7a51c54b658945c0ba0bbb26e5ee1f763bed`. Upstream credit: Juan Lopez Alcaraz
and Nils Strodthoff; S4 code credited by the original repository to
HazyResearch/state-spaces. Retain the root MIT `LICENCE` with redistributed code.

| Local file | Original file / Git blob |
| --- | --- |
| imputer.py | src/imputers/SSSDS4Imputer.py / a6b3892960615b420be528f039752b4dd8d6e827 |
| s4.py | src/imputers/S4Model.py / 94c879208b00a6e6d877319919fc40b5b4fc3721 |
| embedding.py | calc_diffusion_step_embedding from src/utils/util.py / 9921d4eb978747065badad04b247770371b8073a |

`imputer.py`: only the two project-local imports change; an AST test checks the
rest of the file against the preserved upstream source. Capacity and sequence
length are runtime settings, not a replacement network.

`embedding.py`: preserve sinusoidal embedding formula, allocate on the input
device instead of hardcoded CUDA; validate positive even dimension >=4.

`s4.py`: retain original classes and operations. Changes are limited to:

1. Remove unused Lightning/wandb logging and custom CUDA/PyKeOps import setup.
2. Use a native torch Cauchy sum over both stored complex states and their
   conjugates. The original CUDA path sets symmetric=True and the PyKeOps path
   uses conjugates; the original slow fallback omits them. This repairs that
   fallback, not a claim of equality with its buggy output. Tests compare the
   resulting FFT kernel against dense state-space impulse recurrence for N=4/64.
3. Materialize registered state tensors with clone().contiguous(), avoiding
   zero-stride expanded views that modern einops can produce. Values and
   trainability do not change; checkpoint load_state_dict can copy safely.
4. Permit a temporary cached length-24 kernel only during no-grad eval. The
   wrapper clears it after each sampling call, including exception paths.

No alternative Transformer/RNN replaces S4, no adaptive extra trainable graph
is introduced, and upstream default S4 parameter/buffer choices are retained.
No accelerated extension parity or CUDA benchmark was run locally.

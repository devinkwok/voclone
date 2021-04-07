# Background Injection for Noise Separating Voice Conversion GANs

This implementation is built on the U-GAT-IT &mdash; Official PyTorch Implementation.

Ablation samples in `voclone-ablation-examples`.
Samples are randomly drawn from training dataset
Samples are named by ablation model, generator direction, and iteration.
Samples are ordered as noise, real X, identity gen_Y2X(X), generated gen_X2Y(X), cycle identity gen_Y2X(gen_X2Y(X)).
Each sample is 2 seconds long.
Ran out of training time to generate `ablation-geninject-bgdis` samples, however, they should be similar to `ablation-disinject-bgdis`.
The `bgdis` regulariation was probably too high in the ablation runs.

The 'cohen' dataset is under copyright and not included. Synthetic dataset TBD.

Model training has been refactored from original, but need to remove depreciated hyperparameters.
Sequence generation codebase is still a mess, need to remove hard links in `UGATIT.py`.

Most preprocessing scripts are in `voclone-tools`, but some are missing
(e.g. fine-tuned WaveGlow checkpoint and scripts).
All ablation study models are generated using scripts with prefix `ablation-`.

Writeup and presentation to be uploaded at later date.

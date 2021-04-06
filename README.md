# Background Injection for Noise Separating Voice Conversion GANs

This implementation uses U-GAT-IT &mdash; Official PyTorch Implementation.

The 'cohen' dataset is under copyright and thus not available.

Ablation samples in `voclone-ablation-examples`.
Samples are randomly drawn during training
Samples are named by ablation model, generator direction, and iteration.
Samples are ordered as noise, real X, identity gen_Y2X(X), generated gen_X2Y(X), cycle identity gen_Y2X(gen_X2Y(X)).
Each sample is 2 seconds long.
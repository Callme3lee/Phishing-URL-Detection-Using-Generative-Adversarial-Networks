# Interactive URL Detection Explainer

This self-contained walkthrough uses the real browser preprocessing
and the real three ONNX checkpoints for HYB-MS + 10% MS-HardGAN-C.

Run ./start_server.sh from this folder, then open http://127.0.0.1:8765 in
Chrome or Chromium. A local server is required because browsers do not load the
model resources safely from a direct file page.

The explainer shows canonicalisation, the character-CNN input, selected values
from the 35 lexical features, feature fusion, actual seed 42/43/44 scores, their
mean, and the frozen model-only tier. Each stage also shows the corresponding
model mathematics, with a short symbol guide. During a run, the formulas are
filled with the actual URL lengths, token counts, scaled lexical value, seed
scores, ensemble mean, threshold comparison, tier, and final action. Values for
hidden neural activations are not invented; those stages show their true tensor
dimensions instead. Before the numbered pipeline, a URL-anatomy diagram divides
the address into scheme, bracketed hostname parts, port, path, query, and
fragment. The detailed equations are concentrated in feature fusion and the
three-seed scoring calculation. The demonstration ONNX copies expose their
existing pre-sigmoid logits as an additional output, so Step 5 shows the actual
weighted sum, frozen bias, logit, and sigmoid score without changing the model's
probability. The interface follows the extension's visual design. It calculates
the complete pipeline immediately after local model inference finishes, without
step delays, automatic scrolling, or animated transitions. It never opens the
entered address.

The score is a ranking output, not a guaranteed probability. The full extension
may additionally apply reviewed-service policy. MS-HardGAN-C is used offline
for training and is not called when a URL is entered.

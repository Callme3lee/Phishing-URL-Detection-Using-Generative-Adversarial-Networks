# Version 3 Neural Baselines

## Controlled design

CNN, LSTM, and RNN use the same 358,619 scheme-free real training rows, 168-character vocabulary, maximum encoded length of 128, hostname-held-out validation/test splits, seeds 42/43/44, class-weighted binary cross-entropy, AdamW, mixed precision, and early stopping based on validation PR-AUC. Each operating threshold maximizes validation F1 while constraining validation FPR to at most 1%.

## Architectures

- **CNN:** 48-dimensional character embeddings and 128 filters for kernel widths 3, 4, and 5. It detects local tokens and short phishing motifs in parallel.
- **LSTM:** 48-dimensional embeddings and one 96-unit LSTM layer. Its input, forget, and output gates preserve useful dependencies across the URL.
- **RNN:** 48-dimensional embeddings and one 96-unit tanh recurrent layer. It was added as a simpler recurrent control without LSTM gates.

The RNN is intentionally uncomplicated and can be retrained on the laptop GPU. Its weaker and more variable result provides an informative comparison rather than another needlessly large model.

## Conclusion

LSTM is the best overall internal classifier, while CNN gives slightly better recall and ranking performance. Vanilla RNN is competitive but its lower F1 and higher standard deviation indicate that the LSTM gates improve optimization and sequence retention. The result supports retaining CNN and LSTM as the primary Version 3 neural baselines and treating RNN as the architectural control.

All checkpoints and metrics are under `artifacts/baselines/{cnn,lstm,rnn}/seed_{42,43,44}`.

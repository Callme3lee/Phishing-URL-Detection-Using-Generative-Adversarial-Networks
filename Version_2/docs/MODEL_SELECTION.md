# Model Selection and Reasoning

## Baselines

The character CNN and LSTM learn directly from URL text. The CNN is the primary
deployment candidate because its convolutions are parallel and capture local
patterns such as suspicious tokens, subdomain fragments, encodings, and path
structures. The LSTM tests whether sequential order adds value.

Random Forest and linear SVM provide understandable lexical-feature controls.
They test whether neural character models and GAN augmentation add value beyond
length, entropy, digit, delimiter, hostname, path, query, and suspicious-token
statistics.

## Generator search

No GAN is universally “best” for URLs. The selected pair covers the two main
ways of handling discrete text while remaining runnable on the project RTX 3060:

### Gumbel-LSGAN

The generator is an autoregressive LSTM pretrained by maximum likelihood.
Gumbel-Softmax supplies differentiable approximate token samples during
adversarial training. The least-squares discriminator objective is retained
because it supplies smoother non-saturating gradients and was stable in Version
1. This is the practical stability candidate.

### SeqGAN

SeqGAN treats sequence generation as a stochastic policy and uses discriminator
rewards for policy-gradient updates. It directly addresses the fact that hard
discrete token sampling blocks ordinary backpropagation. This is the
text-specific research candidate. Maximum-likelihood regularization and a
limited rollout schedule are used to keep it tractable.

### Why not make RelGAN mandatory?

RelGAN was designed specifically for text and reported a controllable
quality/diversity trade-off, but its relational-memory generator and richer
discriminator substantially increase implementation and tuning complexity. It
was therefore excluded from the primary two-model study. It has since been
trained as a clearly separated exploratory candidate; its checkpoint audit does
not retroactively change the predeclared Gumbel-LSGAN/SeqGAN comparison, and it
remains outside classifier augmentation pending full-corpus verification.

### Why WGAN-GP moved to Version 1

WGAN-GP is valuable for stabilizing continuous GAN critics, but the URL problem
still requires a discrete-token solution. Its gradient penalty is also costly
at sequence length 128 on a 6 GB GPU. Version 2 prioritizes a text-specific
SeqGAN comparison and a simpler differentiable LSGAN.

## Primary references

- Yu et al., [“SeqGAN: Sequence Generative Adversarial Nets with Policy
  Gradient,”](https://aaai.org/papers/10804-aaai-31-2017/) AAAI 2017.
- Mao et al., [“Least Squares Generative Adversarial
  Networks,”](https://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html)
  ICCV 2017.
- Nie et al., [“RelGAN: Relational Generative Adversarial Networks for Text
  Generation,”](https://openreview.net/forum?id=rJedV3R5tm) ICLR 2019.
- Jang et al., [“Categorical Reparameterization with
  Gumbel-Softmax,”](https://openreview.net/forum?id=rkE3y85ee) ICLR 2017.

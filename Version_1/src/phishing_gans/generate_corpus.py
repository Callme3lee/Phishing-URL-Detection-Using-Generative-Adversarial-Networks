from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .autoregressive_models import AutoregressiveURLGenerator, to_base_tokens
from .common import require_cuda, seed_everything
from .train_gan import decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an offline URL corpus from a GAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("--count and --batch-size must be positive")
    device = require_cuda()
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    generator = AutoregressiveURLGenerator(**checkpoint["model_args"]).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    generator.eval()
    base_vocab_size = len(metadata["vocab"])
    batches = []
    with torch.inference_mode():
        for start in range(0, args.count, args.batch_size):
            size = min(args.batch_size, args.count - start)
            noise = torch.randn(size, generator.latent_dim, device=device)
            generated = generator.generate(
                noise,
                bos_id=checkpoint["bos_id"],
                sequence_length=metadata["max_length"],
                temperature=args.temperature,
                differentiable=False,
            )
            batches.append(
                to_base_tokens(
                    generated,
                    base_vocab_size=base_vocab_size,
                    bos_id=checkpoint["bos_id"],
                    eos_id=checkpoint["eos_id"],
                ).cpu().numpy()
            )
            if (start // args.batch_size + 1) % 20 == 0:
                print(f"generated {start + size}/{args.count}", flush=True)
    tokens = np.concatenate(batches, axis=0)
    inverse = {
        value: key for key, value in metadata["vocab"].items() if len(key) == 1
    }
    urls = decode(tokens, inverse)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "raw_tokens.npy", tokens)
    (args.output_dir / "raw_urls.txt").write_text("\n".join(urls), encoding="utf-8")
    generation = {
        "checkpoint": str(args.checkpoint),
        "gan_type": checkpoint.get("gan_type", "unknown"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "count": len(urls),
        "temperature": args.temperature,
        "seed": args.seed,
    }
    (args.output_dir / "generation.json").write_text(
        json.dumps(generation, indent=2), encoding="utf-8"
    )
    print(json.dumps(generation, indent=2))
    print("first samples:")
    for url in urls[:10]:
        print(f"  {url}")


if __name__ == "__main__":
    main()

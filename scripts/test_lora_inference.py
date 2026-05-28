"""
Quick test of a trained style LoRA using the production wan_service pipeline.

Usage:
  PYTHONPATH=. python scripts/test_lora_inference.py \
    --selfie /path/to/selfie.jpg \
    --lora_dir /tmp/stylevid2/loras/test-user-1 \
    --output /tmp/test_lora_out.mp4 \
    --prompt "walking in a city at sunset"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.core.logging_config import setup_logging
setup_logging()
log = logging.getLogger("test_lora")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfie",   required=True, type=Path)
    parser.add_argument("--lora_dir", required=True, type=Path)
    parser.add_argument("--output",   default="/tmp/test_with_lora.mp4", type=Path)
    parser.add_argument("--prompt",   default="walking in a city at sunset, natural movement")
    parser.add_argument("--num_frames",      type=int,   default=17)
    parser.add_argument("--num_steps",       type=int,   default=None, help="inference steps (default: from config, 20)")
    parser.add_argument("--guidance_scale",  type=float, default=5.0)
    parser.add_argument("--lora_scale",      type=float, default=0.8)
    parser.add_argument("--no_lora", action="store_true")
    args = parser.parse_args()

    if not args.selfie.exists():
        raise FileNotFoundError(args.selfie)

    lora_file = args.lora_dir / "lora_weights.safetensors"
    if not lora_file.exists():
        lora_file = args.lora_dir / "lora_weights.pt"

    trigger_word_file = args.lora_dir / "trigger_word.txt"
    trigger_word = trigger_word_file.read_text().strip() if trigger_word_file.exists() else ""

    prompt = args.prompt
    if not args.no_lora and trigger_word and not prompt.startswith(trigger_word):
        prompt = f"{trigger_word} person, {prompt}"
    log.info(f"Prompt: {prompt!r}  lora={'off' if args.no_lora else lora_file.name}")

    from backend.services import wan_service
    wan_service.generate_i2v(
        selfie_path=args.selfie,
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted face",
        output_path=args.output,
        num_frames=args.num_frames,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        lora_path=None if args.no_lora else lora_file,
        lora_scale=args.lora_scale,
    )
    log.info(f"Done → {args.output}")
    log.info(f"Open with: open {args.output}")


if __name__ == "__main__":
    main()

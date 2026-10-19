"""
Quick example to show how device selection can be controlled, and was checked
"""
import argparse
import os
import time
from kokoro import KPipeline
from kokoro.hf_utils import set_checkpoints_dir
from loguru import logger

def generate_audio(pipeline, text):
    for _, _, audio in pipeline(text, voice='af_bella'):
        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"
        return samples

def time_synthesis(device=None, checkpoint_dir=None):
    try:
        start = time.perf_counter()
        pipeline = KPipeline(lang_code='a', device=device, checkpoint_dir=checkpoint_dir)
        samples = generate_audio(pipeline, "The quick brown fox jumps over the lazy dog.")
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"✓ {device or 'auto':<6} | {ms:>5.1f}ms total | {samples:>6,d} samples")
    except RuntimeError as e:
        logger.error(f"✗ {'cuda' if 'CUDA' in str(e) else device or 'auto':<6} | {'not available' if 'CUDA' in str(e) else str(e)}")

def compare_shared_model(checkpoint_dir=None):
    try:
        start = time.perf_counter()
        en_us = KPipeline(lang_code='a', checkpoint_dir=checkpoint_dir)
        en_uk = KPipeline(lang_code='a', model=en_us.model, checkpoint_dir=checkpoint_dir)
        
        for pipeline in [en_us, en_uk]:
            generate_audio(pipeline, "Testing model reuse.")
                
        ms = (time.perf_counter() - start) * 1000
        logger.info(f"✓ reuse  | {ms:>5.1f}ms for both models")
    except Exception as e:
        logger.error(f"✗ reuse  | {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Device Selection & Performance")
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default=os.getenv("CHECKPOINT_DIR", "checkpoints"),
        help="Path to checkpoint directory (default: 'checkpoints' or $CHECKPOINT_DIR)",
    )
    args = parser.parse_args()
    set_checkpoints_dir(args.checkpoint_dir)

    logger.info("Device Selection & Performance")
    logger.info("-" * 40)
    time_synthesis(checkpoint_dir=args.checkpoint_dir)
    time_synthesis('cuda', checkpoint_dir=args.checkpoint_dir)
    time_synthesis('cpu', checkpoint_dir=args.checkpoint_dir) 
    logger.info("-" * 40)
    compare_shared_model(checkpoint_dir=args.checkpoint_dir)
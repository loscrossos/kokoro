import os
import torch
from kokoro import KPipeline

try:
    import soundfile as sf
except ImportError:
    sf = None


def test_german_inference(checkpoint_dir=None):
    pipeline = KPipeline(lang_code='d', checkpoint_dir=checkpoint_dir)
    text = "Kokoro German ist ein Text to Speech Modell mit zweiundachtzig Millionen Parametern."

    # Create a temporary mock voice tensor for CI/unit testing (isolated offline execution)
    mock_voice_path = 'voices/mock_german_voice.pt'
    os.makedirs('voices', exist_ok=True)
    
    mock_tensor = torch.randn(510, 1, 256)
    torch.save(mock_tensor, mock_voice_path)

    try:
        # Force the framework to evaluate our newly placed mock voice file off local disk
        generator = pipeline(text, voice=mock_voice_path, speed=1.0)
        for i, (gs, ps, audio) in enumerate(generator):
            assert audio is not None and len(audio) > 0, "Audio generation failed"
            # sf.write(f"german_test_output_{i}.wav", audio, 24000) # skip disk write in CI
        print("CI Test Passed: Generated audio successfully using isolated mock German tensors.")
    finally:
        # Clean up the temporary fixture upon test completion
        if os.path.exists(mock_voice_path):
            os.remove(mock_voice_path)
        if os.path.exists('voices') and not os.listdir('voices'):
            os.rmdir('voices')


if __name__ == "__main__":
    import argparse
    from kokoro.hf_utils import set_checkpoints_dir

    parser = argparse.ArgumentParser(description="German Inference Test")
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default=os.getenv("CHECKPOINT_DIR", "checkpoints"),
        help="Path to checkpoint directory (default: 'checkpoints' or $CHECKPOINT_DIR)",
    )
    args = parser.parse_args()
    set_checkpoints_dir(args.checkpoint_dir)
    test_german_inference(checkpoint_dir=args.checkpoint_dir)

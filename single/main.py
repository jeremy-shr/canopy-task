import argparse
import os
import sys

from pipeline import process_wav


def main():
    parser = argparse.ArgumentParser(
        description="Transcription pipeline: denoise, diarize, and transcribe a WAV file."
    )
    parser.add_argument("wav_path", help="Path to input WAV file")
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for output files (default: ./output)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.wav_path):
        print(f"Error: WAV file not found: {args.wav_path}")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required for pyannote.")
        print("Export it with: export HF_TOKEN=your_token_here")
        sys.exit(1)

    process_wav(
        wav_path=args.wav_path,
        output_dir=args.output_dir,
        hf_token=hf_token,
        device=args.device,
    )


if __name__ == "__main__":
    main()

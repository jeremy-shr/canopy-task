import argparse
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single"))

from datasets import concatenate_datasets
from pipeline import process_wav


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcription pipeline: process a folder of WAV files."
    )
    parser.add_argument("input_dir", help="Directory containing WAV files")
    parser.add_argument(
        "--output-dir",
        default="./batch_output",
        help="Output directory (default: ./batch_output)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers per file (default: auto-detect)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required for pyannote.")
        sys.exit(1)

    wav_files = sorted(glob(os.path.join(args.input_dir, "*.wav")))
    if not wav_files:
        print(f"No WAV files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV files")

    all_datasets = []
    for i, wav_path in enumerate(wav_files):
        stem = Path(wav_path).stem
        file_output_dir = os.path.join(args.output_dir, stem)

        # Skip already processed files
        dataset_path = os.path.join(file_output_dir, "dataset")
        if os.path.exists(dataset_path):
            print(f"\n[{i+1}/{len(wav_files)}] Skipping {stem} (already processed)")
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)
            ds = ds.add_column("Source", [stem] * len(ds))
            all_datasets.append(ds)
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(wav_files)}] Processing: {Path(wav_path).name}")
        print(f"{'='*60}")

        ds = process_wav(
            wav_path=wav_path,
            output_dir=file_output_dir,
            hf_token=hf_token,
            device=args.device,
            num_speakers=args.num_speakers,
        )
        ds = ds.add_column("Source", [stem] * len(ds))
        all_datasets.append(ds)

    print(f"\nCombining {len(all_datasets)} datasets...")
    combined = concatenate_datasets(all_datasets)

    combined_path = os.path.join(args.output_dir, "combined_dataset")
    combined.save_to_disk(combined_path)
    print(f"Combined dataset saved to {combined_path} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()

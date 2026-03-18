# Canopy Transcription Pipeline

Audio transcription pipeline that takes a WAV file with multiple speakers and background noise, and produces a HuggingFace dataset with denoised audio segments, transcripts, and speaker labels.

## Pipeline

1. **Denoise** — Removes background noise using Demucs (htdemucs model)
2. **Diarize** — Identifies speakers and their time segments using PyAnnote
3. **Slice + Transcribe** — Splits audio by speaker segment and transcribes each with faster-whisper
4. **Build Dataset** — Assembles a HuggingFace dataset with columns: `Transcript`, `Audio`, `Speaker`

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU
- HuggingFace account with accepted model terms

### Accept PyAnnote model terms

Visit and accept the terms on both:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### Install dependencies

```bash
pip install -r single/requirements.txt
```

### Set HuggingFace token

```bash
export HF_TOKEN="hf_your_token_here"
```

## Usage

### Single file

```bash
cd single
python main.py /path/to/file.wav --output-dir ./output --num-speakers 2
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `wav_path` | Yes | — | Path to input WAV file |
| `--output-dir` | No | `./output` | Directory for output files |
| `--device` | No | `cuda:0` | CUDA device to use |
| `--num-speakers` | No | auto-detect | Number of speakers in the audio |

### Output

```
output/
├── <filename>_denoised.wav    # Denoised audio
├── segments/                  # Individual speaker segments
│   ├── <filename>_seg0000.wav
│   ├── <filename>_seg0001.wav
│   └── ...
└── dataset/                   # HuggingFace dataset
```

The dataset has three columns:

| Column | Type | Description |
|---|---|---|
| `Transcript` | string | Transcribed text for the segment |
| `Audio` | Audio | Denoised audio for the segment |
| `Speaker` | string | Speaker label (e.g. SPEAKER_00) |

### Batch processing

Process all WAV files in a directory:

```bash
cd batch
python main.py /path/to/wav/folder --output-dir ./batch_output --num-speakers 2
```

#### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `input_dir` | Yes | — | Directory containing WAV files |
| `--output-dir` | No | `./batch_output` | Output directory |
| `--device` | No | `cuda:0` | CUDA device to use |
| `--num-speakers` | No | auto-detect | Number of speakers per file |

- Already-processed files are skipped on rerun
- Each file is processed into its own subdirectory, then all results are combined into a single dataset at `<output-dir>/combined_dataset/`

#### Output

```
batch_output/
├── file1/                     # Per-file output
│   ├── file1_denoised.wav
│   ├── segments/
│   └── dataset/
├── file2/
│   └── ...
└── combined_dataset/          # Combined dataset across all files
```

The combined dataset has four columns:

| Column | Type | Description |
|---|---|---|
| `Transcript` | string | Transcribed text for the segment |
| `Audio` | Audio | Denoised audio for the segment |
| `Speaker` | string | Speaker label (e.g. SPEAKER_00) |
| `Source` | string | Source WAV filename (without extension) |

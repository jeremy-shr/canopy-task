import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, Dataset
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline


def denoise(wav_path: str, output_dir: str, device: str = "cuda:0") -> str:
    """Remove background noise using Demucs. Returns path to denoised vocals."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs")
    model.to(device)

    wav, sr = torchaudio.load(wav_path)
    # apply_model expects (batch, channels, samples)
    wav = wav.unsqueeze(0).to(device)

    sources = apply_model(model, wav)
    # sources shape: (batch, num_sources, channels, samples)
    # htdemucs sources: drums, bass, other, vocals
    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx].cpu()

    stem_name = Path(wav_path).stem
    out_path = os.path.join(output_dir, f"{stem_name}_denoised.wav")
    torchaudio.save(out_path, vocals, sr)

    return out_path


def diarize(audio_path: str, hf_token: str, device: str = "cuda:0") -> list[dict]:
    """Run speaker diarization. Returns list of {start, end, speaker}."""
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(torch.device(device))

    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    return segments


def slice_audio(audio_path: str, start: float, end: float) -> tuple[np.ndarray, int]:
    """Extract a segment from an audio file. Returns (audio_array, sample_rate)."""
    audio, sr = sf.read(audio_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample], sr


def transcribe(audio_array: np.ndarray, sr: int, model: WhisperModel) -> str:
    """Transcribe an audio segment using faster-whisper."""
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    segments, _ = model.transcribe(audio_array, language="en")
    return " ".join(seg.text.strip() for seg in segments)


def build_dataset(rows: list[dict]) -> Dataset:
    """Build a HuggingFace Dataset with Transcript, Audio, Speaker columns."""
    ds = Dataset.from_dict({
        "Transcript": [r["transcript"] for r in rows],
        "Audio": [r["audio_path"] for r in rows],
        "Speaker": [r["speaker"] for r in rows],
    })
    ds = ds.cast_column("Audio", Audio())
    return ds


def process_wav(
    wav_path: str,
    output_dir: str,
    hf_token: str,
    device: str = "cuda:0",
) -> Dataset:
    """Full pipeline: denoise -> diarize -> slice -> transcribe -> dataset."""
    os.makedirs(output_dir, exist_ok=True)
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    print(f"[1/4] Denoising {wav_path}...")
    clean_path = denoise(wav_path, output_dir, device=device)

    print("[2/4] Diarizing...")
    segments = diarize(clean_path, hf_token, device=device)
    print(f"      Found {len(segments)} segments")

    print("[3/4] Transcribing segments...")
    device_name = device.split(":")[0]
    device_index = int(device.split(":")[1]) if ":" in device else 0
    whisper_model = WhisperModel(
        "large-v3",
        device=device_name,
        device_index=device_index,
        compute_type="float16",
    )

    rows = []
    stem = Path(wav_path).stem
    for i, seg in enumerate(segments):
        audio_chunk, sr = slice_audio(clean_path, seg["start"], seg["end"])

        if len(audio_chunk) / sr < 0.3:
            continue

        seg_path = os.path.join(segments_dir, f"{stem}_seg{i:04d}.wav")
        sf.write(seg_path, audio_chunk, sr)

        text = transcribe(audio_chunk, sr, whisper_model)

        rows.append({
            "transcript": text,
            "audio_path": seg_path,
            "speaker": seg["speaker"],
        })
        print(
            f"      [{i+1}/{len(segments)}] {seg['speaker']} "
            f"({seg['start']:.1f}s-{seg['end']:.1f}s): {text[:60]}..."
        )

    print("[4/4] Building dataset...")
    ds = build_dataset(rows)

    ds_path = os.path.join(output_dir, "dataset")
    ds.save_to_disk(ds_path)
    print(f"      Dataset saved to {ds_path} ({len(ds)} rows)")

    return ds

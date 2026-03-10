# ASR & Diarization Pipeline

This Repo contains the audio processing pipeline for running of the AISIC project. The pipeline is the result of various experiments and is meant to be run on a local computer. It performs speaker diarization and automatic speech recognition and creates transcription files.

## What's Included

- **Whisper ASR**: Transcribes audio files in German (configurable) using OpenAI's Whisper model
- **Speaker Diarization**: Uses NVIDIA NeMo to identify and separate different speakers
- **Timestamp Alignment**: Aligns word-level timestamps using Wav2Vec
- **Multi-speaker JSON Output**: Combines transcriptions with speaker labels and precise timing
- **Multi-Audiotrack Handling**: Transcribe interview consisting of multiple audio tracks as one transcribt (online interview setting)

## Usage

1. **Edit configuration** in [src/asr_pipeline.py](src/asr_pipeline.py):
   - Set `input_dir` (audio files location)
   - Set `output_dir` (where results are saved)
   - Set `whisper_model` (e.g., "turbo", "large")
   - Set `language` (default: "de" for German)
   - Adjust ASR parameters as needed (beam_size, temperature, etc.)
   - Configure device settings (GPU index, batch size)

2. **Run from the main directory**:
   ```bash
   conda activate transscription
   python src/asr_pipeline.py
   # for multiple audio tracks -> python src/asr_pipeline_split.py
   ```

In case you have multiple audio tracks in one multimedia container (mkv/mp4) belonging to one interview also check the preprocessing script `src/split_tracks.py`.

    If problems appear, run the following code before executing the script:
   ```bash
   export LD_LIBRARY_PATH=/home/your_user_name/miniconda3/envs/trans2/lib/python3.10/site-packages/nvidia/cudnn/lib
   ```


The pipeline processes all audio files in `input_dir`, skipping files that are already transcribed. Results are saved as JSON files in `output_dir`.


import whisperx

def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    asr_options: dict,
    device: str,
):
    # Faster Whisper batched
    whisper_model = whisperx.load_model(
        model_name,
        "cuda",
        #device_index=1,
        compute_type=compute_dtype,
        asr_options=asr_options,
        vad_method="silero"
    )

    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)
    return result["segments"], result["language"]

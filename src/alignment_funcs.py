import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
import torch

def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
    # if current word is the last word
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]
 
    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            # if next word doesn't have a start timestamp
            # merge it with the current word and delete it
            word_timestamps[current_word_index]["word"] += (
                " " + word_timestamps[next_word_index]["word"]
            )
 
            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1
            if next_word_index == len(word_timestamps):
                return final_timestamp
 
        else:
            return word_timestamps[next_word_index]["start"]

def filter_missing_timestamps(
    word_timestamps, initial_timestamp=0, final_timestamp=None
):
    # handle the first and last word
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = (
            initial_timestamp if initial_timestamp is not None else 0
        )
        word_timestamps[0]["end"] = _get_next_start_timestamp(
            word_timestamps, 0, final_timestamp
        )
 
    result = [
        word_timestamps[0],
    ]
 
    for i, ws in enumerate(word_timestamps[1:], start=1):
        # if ws doesn't have a start and end
        # use the previous end as start and next start as end
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)
 
        if ws["word"] is not None:
            result.append(ws)
    return result

def align_whisper(whisper_results, audio_path, language, batch_size, device):
    wav2vec2_langs = list(DEFAULT_ALIGN_MODELS_TORCH.keys()) + list(
        DEFAULT_ALIGN_MODELS_HF.keys()
    )

    whisper_langs = sorted(LANGUAGES.keys()) + sorted(
        [k.title() for k in TO_LANGUAGE_CODE.keys()]
    )

    if language in wav2vec2_langs:
        device = "cuda"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, audio_path, device
        )
        word_timestamps = filter_missing_timestamps(
            result_aligned["word_segments"],
            initial_timestamp=whisper_results[0].get("start"),
            final_timestamp=whisper_results[-1].get("end"),
        )

        # clear gpu vram
        del alignment_model
        torch.cuda.empty_cache()
    else:
        assert batch_size == 0, (  # TODO: add a better check for word timestamps existence
            f"Unsupported language: {language}, use --batch_size to 0"
            " to generate word timestamps using whisper directly and fix this error."
        )
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})

    # -------------------------------
    # Fix timestamp bleeding 
    # -------------------------------
    def fix_whisper_timestamps(words, max_word_duration=2.0):
        """
        Fixes Whisper timestamp bleeding.
        - Prevents word end spilling into next word
        - Caps absurd word durations
        """
        fixed = []

        for i, w in enumerate(words):
            start = w["start"]
            end = w["end"]

            # Rule 1: cannot exceed next word start
            if i < len(words) - 1:
                next_start = words[i + 1]["start"]
                end = min(end, next_start)

            # Rule 2: cap absurd durations
            if end - start > max_word_duration:
                end = start + max_word_duration

            fixed.append({
                **w,
                "start": start,
                "end": end
            })

        return fixed

    word_timestamps = fix_whisper_timestamps(word_timestamps)

    return word_timestamps
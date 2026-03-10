import os, json
import whisper_funcs
import alignment_funcs
import nvidia_nemo_funcs
import combine_diar_trans
import audio_preprocessing
import interview_processing
import transcript_merge

# =========================
# Meta config
# =========================
device = "cuda"
device_index = 1
batch_size = 16
compute_type = "float16"

input_dir_split = "/media/veracrypt1/test/"
output_dir = "/media/veracrypt1/transcripts_test/"
temp_dir = "./nemo_tmp"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mkv"}

# =========================
# ASR config
# =========================

whisper_model = "turbo"
language = "de"
asr_config =  {
    "beam_size": 5, # verhindert halluzinationen, macht aber logits schwieriger?
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": True,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": "Und gibt ja auch ja weiß ich net ob das auch so bisschen erziehungssache ist oder wie mans auch vorlebt ne, manche sind ja da auch ähm schneller anfällig oder auch ähm bes vielleicht auch psychisch net so belastbar ähm gehört ja auch alles so mit rein.  Mmmh (..) Vielen lieben Dank! Ähm dann würde ich jetzt noch gerne wissen, und wie würden Sie ihren eigenen Gesundheitszustand im Allgemeinen beschreiben?",
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "suppress_numerals": True,
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
    "hotwords": None
}

# Seperate Audio Tracks, Speakers from Filename
# IMPORTANT: run split_tracks.py before executing this
for filename in os.listdir(input_dir_split):
    print(filename)
    audio_file = os.path.join(input_dir_split, filename)
    output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

    # Check if file has valid audio extension

    if os.path.splitext(audio_file)[1].lower() not in AUDIO_EXTS:
        continue

    # Check if interview has been transcribed before

    if os.path.exists(output_file):
        print(f"{output_file} is already transcribed.")
        continue

    # # already resampled to 16kHz when exporting seperate audio tracks
    #print("Converting...")
    #resample to 16kHz and convert to mono
    #audio_preprocessing.resample_mono(audio_file)


    try:
        print("Transcribing...")
        whisper_results, language = whisper_funcs.transcribe_batched(
            audio_file,
            language,
            batch_size,
            whisper_model,
            compute_type,
            asr_config,
            device,
        )

        print(whisper_results)
        print("Aligning timestamps...")
        word_timestamps = alignment_funcs.align_whisper(whisper_results, audio_file, language, batch_size, device)

        # Infer speaker from filename
        speaker_id = interview_processing.infer_speaker_from_filename(filename)
        print(f"Speaker inferred: {speaker_id}")

        interview = interview_processing.aggregate_to_speech_acts(word_timestamps, pause_threshold=0.5)
        # interview = word_timestamps

        for speech_act in interview:
            speech_act["speaker"] = speaker_id

        with open(output_file, "w", encoding="utf-8") as s:
            json.dump(interview, s, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Failed on {filename}: {e}")


# Merging and transformation to ASS

combined_dir="/media/veracrypt1/transcripts_merged/"
transcript_merge.merge_interview_transcripts(input_dir=output_dir, output_dir=combined_dir, audio_dir=input_dir_split)

for filename in os.listdir(combined_dir):

    filepath = os.path.join(combined_dir, filename)
    out_ass = os.path.join(combined_dir, os.path.splitext(filename)[0] + ".ass")
    interview_processing.json_to_ass(filepath, out_ass)

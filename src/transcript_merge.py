import os
import json
from collections import defaultdict
import subprocess

# =========================
# Internal helpers
# =========================

def _get_audio_duration(audio_path):
    """
    Returns the total duration of an audio/video file in seconds.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return float(result.stdout.strip())

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _group_transcripts(input_dir):
    """
    Groups transcripts by interview ID.

    Returns:
        dict:
        {
            interview_id: {
                "interviewer": filename,
                "parent": filename,
                "kid": filename
            }
        }
    """
    groups = defaultdict(dict)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        base = filename[:-5]  # remove .json

        if "_interviewer" in base:
            interview_id = base.replace("_interviewer", "")
            groups[interview_id]["interviewer"] = filename

        elif "_interviewee-parent" in base:
            interview_id = base.replace("_interviewee-parent", "")
            groups[interview_id]["parent"] = filename

        elif "_interviewee-kid" in base:
            interview_id = base.replace("_interviewee-kid", "")
            groups[interview_id]["kid"] = filename

    return groups


# =========================
# Public API
# =========================

def merge_interview_transcripts(input_dir, output_dir, audio_dir, verbose=True):
    """
    Merges interviewer + parent + kid transcripts into one timeline.

    Parameters:
        input_dir: directory with individual JSON transcripts
        output_dir: directory to save merged JSONs
        audio_dir: directory with actual audio files for offset calculation
    """
    os.makedirs(output_dir, exist_ok=True)
    groups = _group_transcripts(input_dir)

    merged_count = 0
    skipped_count = 0

    for interview_id, files in groups.items():
        required = {"interviewer", "parent", "kid"}
        if not required.issubset(files.keys()):
            skipped_count += 1
            if verbose:
                print(f"Skipping incomplete interview: {interview_id}")
            continue

        if verbose:
            print(f"Merging interview: {interview_id}")

        # Load JSON transcripts
        interviewer = _load_json(os.path.join(input_dir, files["interviewer"]))
        parent = _load_json(os.path.join(input_dir, files["parent"]))
        kid = _load_json(os.path.join(input_dir, files["kid"]))

        # Compute parent duration from audio file
        parent_audio_file = None
        # Look for matching audio file: same base name
        for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mkv"]:
            candidate = os.path.join(audio_dir, f"{interview_id}_interviewee-parent{ext}")
            if os.path.exists(candidate):
                parent_audio_file = candidate
                break

        if not parent_audio_file:
            print(f"Warning: parent audio not found for {interview_id}, using last segment end as fallback")
            parent_duration = max(seg["end"] for seg in parent) if parent else 0
        else:
            parent_duration = _get_audio_duration(parent_audio_file)

        # -------- OFFSET KID TRACK --------
        for seg in kid:
            seg["start"] += parent_duration
            seg["end"] += parent_duration

        # -------- MERGE --------
        merged = interviewer + parent + kid
        merged.sort(key=lambda x: x["start"])

        # -------- SAVE --------
        out_path = os.path.join(output_dir, interview_id + ".json")
        _save_json(merged, out_path)
        merged_count += 1

    if verbose:
        print("\nMerge finished:")
        print("Merged:", merged_count)
        print("Skipped:", skipped_count)

        

        

def get_audio_duration(audio_path):
    """
    Returns the total duration of an audio/video file in seconds.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return float(result.stdout.strip())

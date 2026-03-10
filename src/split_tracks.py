import os
import subprocess
import shutil

input_dir = "/media/veracrypt1/"
split_dir = os.path.join(input_dir, "split")

os.makedirs(split_dir, exist_ok=True)


def count_audio_tracks(filepath):
    """Return number of audio streams using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            filepath
        ],
        capture_output=True,
        text=True,
        check=True
    )

    lines = result.stdout.strip().splitlines()
    return len(lines)


def extract_tracks(input_file):
    """Split interviewer / interviewees tracks."""
    parts = ["interviewer", "interviewees"]

    for track_index in [1, 2]:
        output_file = f"{os.path.splitext(input_file)[0]}_{parts[track_index - 1]}.mkv"

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", input_file,
            "-map", f"0:a:{track_index}",
            "-c", "copy",
            output_file
        ], check=True)


# ---- MAIN LOOP ----

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".mp4", ".mkv")):
        continue

    full_path = os.path.join(input_dir, filename)

    if not os.path.isfile(full_path):
        continue

    try:
        audio_tracks = count_audio_tracks(full_path)

        print(f"{filename}: {audio_tracks} audio tracks")

        if audio_tracks <= 1:
            # do nothing
            continue

        # Move to split folder
        new_path = os.path.join(split_dir, filename)
        shutil.move(full_path, new_path)

        # Split tracks
        extract_tracks(new_path)

    except Exception as e:
        print(f"Error processing {filename}: {e}")

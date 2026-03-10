import pysubs2
import json
import os

def aggregate_to_speech_acts(
    word_timestamps,
    pause_threshold=2.0,
):
    speech_acts = []

    current = {
        "start": word_timestamps[0]["start"],
        "end": word_timestamps[0]["end"],
        "text": word_timestamps[0]["word"],
    }

    for prev, curr in zip(word_timestamps, word_timestamps[1:]):
        pause = curr["start"] - prev["end"]

        if pause >= pause_threshold:
            speech_acts.append(current)
            current = {
                "start": curr["start"],
                "end": curr["end"],
                "text": curr["word"],
            }
        else:
            current["text"] += " " + curr["word"]
            current["end"] = curr["end"]

    speech_acts.append(current)
    return speech_acts

def infer_speaker_from_filename(filename: str) -> str:
    """
    Infers the speaker ID from the filename.
    Example:
        I1FID4KID41_interviewer -> I1
        I1FID4KID41_interviewee_parent -> FID4
        I1FID4KID41_interviewee_kid -> KID41
    """
    base_name = os.path.splitext(filename)[0]  # remove extension
    parts = base_name.split("_")  # split by underscore
    if len(parts) < 2:
        return "UNKNOWN"

    ids_part, role = parts[0], parts[1]

    # Extract IDs
    import re
    match = re.match(r"(I\d+)(FID\d+)(KID\d+)", ids_part)
    print("match", match.groups())
    if not match:
        return "UNKNOWN"

    interviewer_id, parent_id, kid_id = match.groups()

    if role == "interviewer":
        return interviewer_id
    elif role == "interviewee-parent":
        return parent_id
    elif role == "interviewee-kid":
        return kid_id
    else:
        return "UNKNOWN"

def json_to_ass(json_file, ass_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    subs = pysubs2.SSAFile()

    for seg in data: #data.get("segments", data):
        start_ms = int(float(seg["start"]) * 1000)
        end_ms = int(float(seg["end"])* 1000)
        text = seg.get("text", "").replace("\n", " ").strip()
        actor = seg.get("speaker", "")
        subs.append(
            pysubs2.SSAEvent(
                start=start_ms,
                end=end_ms,
                text=text,
                name=actor,
                style="Default"
            )
        )

    subs.save(ass_file)

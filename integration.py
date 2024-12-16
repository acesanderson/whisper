"""
This integrates two transcripts generated from a single mp3 file:
    (1) the time stamped transcript from whisper.py
    (2) the diarization timestamps from diarization.py
Into a fully annotated transcript.
"""


def merge_diarization_and_transcript(diarization_data, transcript_data):
    # Create a list of speaker segments with start/end times
    speaker_segments = []
    for line in diarization_data:
        # Parse line like: "SPEAKER dr_visit_12-16-2024 1 1045.522 1.181 <NA> <NA> SPEAKER_02 <NA> <NA>"
        parts = line.split()
        start_time = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        speaker_segments.append(
            {"start": start_time, "end": start_time + duration, "speaker": speaker}
        )

    # For each transcript segment, find the matching speaker
    annotated_transcript = []
    for segment in transcript_data:
        # Parse segment like: "[0.00 -> 0.12]: Do"
        start = float(segment.start)
        end = float(segment.end)
        text = segment.text

        # Find matching speaker segment
        speaker = None
        for spk in speaker_segments:
            if start >= spk["start"] and end <= spk["end"]:
                speaker = spk["speaker"]
                break

        annotated_transcript.append(
            {"start": start, "end": end, "speaker": speaker, "text": text}
        )

    return annotated_transcript


if __name__ == "__main__":
    diarization_data_file = "audio.rttm"
    with open(diarization_data_file, "r") as f:
        diarization_data = f.read()
    transcript_data_file = "processed/dr_visit_12-16-2024_timestamps.txt"
    with open(transcript_data_file, "r") as f:
        transcript_data = f.read()
    integrated = merge_diarization_and_transcript(diarization_data, transcript_data)
    print(integrated)

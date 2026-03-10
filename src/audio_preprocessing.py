from pydub import AudioSegment
from scipy.io.wavfile import write
import numpy as np

def resample_mono(audio_path):

    audio = AudioSegment.from_file(audio_path)

    # Convert to mono
    audio = audio.set_channels(1)

    # Resample to 16 kHz
    audio = audio.set_frame_rate(16000)

    # Convert to NumPy array
    samples = np.array(audio.get_array_of_samples())

    # Write to new WAV file
    output_path = audio_path
    write(output_path, 16000, samples)

    print(f"Saved converted audio to {output_path}")


import os
import glob
from pydub import AudioSegment

max = 6
for file in glob.glob("Normal/*.wav"):
    audio = AudioSegment.from_file(file)
    if(audio.duration_seconds > max):
        max = audio.duration_seconds
        max_name = file
print(max_name)
print(max)

import os
from pydub import AudioSegment
import json


for audioname in os.listdir("Audios"):
    
    filename = "Audios/" + audioname
    print(filename)

    
       
    audio = AudioSegment.from_file(filename)

    # Define start and end times in milliseconds
    start_time = 135 * 1000  # 10 seconds
    end_time = 170 * 1000    # 20 seconds

    # Extract the sub-audio
    sub_audio = audio[start_time:end_time]

    # Save the result
    sub_audio.export("new_" + audioname , format="wav")
    


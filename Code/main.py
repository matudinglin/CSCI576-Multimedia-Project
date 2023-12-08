import json
import sys
import pickle
import cv2

import numpy as np
from scipy.io.wavfile import read

from motion_vector import compute_motion_vector
from set_db import create_constellation
from file_matching import print_match, open_video_player
from scenedetect import detect, ContentDetector

audio_db_path = "./audio_db.json"
shot_db_path = "./shot_db.pkl"

if __name__ == "__main__":
    # Load the scenes from binary files
    with open(shot_db_path, 'rb') as f:
        split_scenes = pickle.load(f)
        
    query_video = sys.argv[1]
    query_scene_list = detect(query_video, ContentDetector(min_scene_len=15))
    
    if len(query_scene_list) == 0:
        print(f'Query video has no scene detected, use audio matching')
        with open(audio_db_path, 'r') as file:
            db = json.load(file)
        Fs, song = read(sys.argv[2])
        song = np.transpose(np.transpose(song)[0])
        input = create_constellation(song, Fs)
        print_match(sys.argv[2], input, db)
    
    else:
        print(f'Query video has scene detected, use motion vector matching')
        frame_old = query_scene_list[0][1].get_frames() - 1
        frame_new = query_scene_list[0][1].get_frames()
        cap = cv2.VideoCapture(query_video)
        motion_vector = compute_motion_vector(cap, frame_old, frame_new)
        motion_vector2 = compute_motion_vector(cap, frame_old-4, frame_new)
        for video, shots in split_scenes.items():
            for shot in shots:
                diff = None
                diff2 = None
                if motion_vector.shape == shot[0].shape:
                    diff = motion_vector - shot[0]
                if motion_vector2.shape == shot[1].shape:
                    diff2 = motion_vector2 - shot[1]
                if diff is None or diff2 is None:
                    continue
                if (diff < 0.00001).all() and (diff2 < 0.00001).all():
                    frame = shot[2]-frame_old-1
                    print(f'Query video is from {video} with frame {frame}')
                    open_video_player(video, frame)

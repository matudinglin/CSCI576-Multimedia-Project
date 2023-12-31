import pickle
import cv2

from scenedetect import detect, ContentDetector
from motion_vector import compute_motion_vector


video_root = './Data/Videos/'
NUM_VIDEOS = 20

# Initial the dictionary
split_scenes = {}
video_paths = []
for i in range(1, NUM_VIDEOS + 1):
    split_scenes[f'video{i}.mp4'] = []
    video_paths.append(video_root + f'video{i}.mp4')

# Get the scenes from the videos
for i, video_path in enumerate(video_paths):
    print(f"Processing video{i+1}...")
    scene_list = detect(video_path, ContentDetector(min_scene_len=15))
    cap = cv2.VideoCapture(video_path)
    # For all scene except the last one
    for scene in scene_list[:-1]:
        frame_old, frame_new = scene[1].get_frames()-1, scene[1].get_frames()
        motion_vector = compute_motion_vector(cap, frame_old, frame_new)
        motion_vector2 = compute_motion_vector(cap, frame_old-4, frame_new)
        if motion_vector is not None and motion_vector2 is not None:
            split_scenes[video_path.split('/')[-1]].append((motion_vector, motion_vector2, frame_old))
            

# Save the scenes to binary files
with open('./Data/split_scenes.pkl', 'wb') as f:
    pickle.dump(split_scenes, f)



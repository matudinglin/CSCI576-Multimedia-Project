from scenedetect import detect, ContentDetector
import pickle

video_root = './Data/Queries/'
NUM_VIDEOS = 11

# Initial the dictionary
split_scenes = {}
video_paths = []
for i in range(1, NUM_VIDEOS + 1):
    print(f"Processing video {i}...")
    split_scenes[f'video{i}_1.mp4'] = []
    video_paths.append(video_root + f'video{i}_1.mp4')

# Get the scenes from the videos
for video_path in video_paths:
    scene_list = detect(video_path, ContentDetector())
    for scene in scene_list:
        split_scenes[video_path.split('/')[-1]].append((scene[0].get_frames(), scene[1].get_frames()))

# Save the scenes to binary files
with open('./Data/split_scenes.pkl', 'wb') as f:
    pickle.dump(split_scenes, f)

# Load the scenes from binary files
with open('./Data/split_scenes.pkl', 'rb') as f:
    split_scenes = pickle.load(f)

print(split_scenes)
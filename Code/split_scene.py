from scenedetect import detect, ContentDetector
import pickle
import cv2

video_root = './Data/Videos/'
NUM_VIDEOS = 20

def compute_motion_vector(cap, frame_old, frame_new):
    # Read the frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_old)
    ret, frame_old = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_new)
    ret, frame_new = cap.read()
    
    # Convert to grayscale
    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    # Detect features to track
    points1 = cv2.goodFeaturesToTrack(gray_old, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Compute optical flow (Lucas-Kanade)
    # Catch the error when there is no feature to track
    try:
        points2, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, points1, None)
        # Calculate motion vectors
        motion_vector = points2 - points1
    except:
        motion_vector = None

    # Calculate the average motion vector
    return motion_vector

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
        if motion_vector is not None:
            split_scenes[video_path.split('/')[-1]].append((motion_vector, frame_old))
            

# Save the scenes to binary files
with open('./Data/split_scenes.pkl', 'wb') as f:
    pickle.dump(split_scenes, f)



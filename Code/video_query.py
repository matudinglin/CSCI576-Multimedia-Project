import pickle
from scenedetect import detect, ContentDetector
import cv2

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
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, points1, None)

    # Calculate motion vectors
    motion_vector = points2 - points1

    # Calculate the average motion vector
    return motion_vector

# Load the scenes from binary files
with open('./Data/split_scenes.pkl', 'rb') as f:
    split_scenes = pickle.load(f)

# # Get the scenes from the query video
# query_video = './Data/Queries/video_1.mp4'
# query_scene_list = detect(query_video, ContentDetector(min_scene_len=15))

# # Get the frames of the scenes from the query video
# frame_old = query_scene_list[0][1].get_frames() - 1
# frame_new = query_scene_list[0][1].get_frames()

# # Compute the motion vector
# cap = cv2.VideoCapture(query_video)
# motion_vector = compute_motion_vector(cap, frame_old, frame_new)

# # Compare motion vector with other motion vectors
# for video, scenes in split_scenes.items():
#     for scene in scenes:
#         # Compare the motion vector with the motion vector of the scene
#         if motion_vector.shape == scene.shape:
#             diff = motion_vector - scene
#             # If the difference is small enough, print the video name
#             if (diff < 0.1).all():
#                 print(video)

# Compare for all query videos
query_video_path = './Data/Queries/video{}_1.mp4'
for i in range(1, 12):
    query_video = query_video_path.format(i)
    query_scene_list = detect(query_video, ContentDetector(min_scene_len=15))
    if len(query_scene_list) == 0:
        print(f'Query video{i}_1 has no scene detected')
        continue
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
                print(f'Query video{i}_1 is from {video} with frame {shot[2]-frame_old-1}')
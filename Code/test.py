from scenedetect import detect, AdaptiveDetector

# Get all the query videos paths
query_videos = []
query_video_path = './Data/Queries/'
for i in range(1, 12):
    query_videos.append(query_video_path + f'video{i}_1.mp4')

# Get all the dataset videos paths
dataset_videos = []
dataset_video_path = './Data/Videos/'
for i in range(1, 21):
    dataset_videos.append(dataset_video_path + f'video{i}.mp4')

for query_video in query_videos:
    query_scene_list = detect(query_video, AdaptiveDetector(min_scene_len=15, luma_only=True, min_content_val = 1))
    print(f'============== For query video {query_video.split("/")[-1]} ==============')
    for i, scene in enumerate(query_scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
            i+1,
            scene[0].get_timecode(), scene[0].get_frames(),
            scene[1].get_timecode(), scene[1].get_frames(),))
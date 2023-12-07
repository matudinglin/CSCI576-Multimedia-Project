from scenedetect import detect, ContentDetector

scene_list = detect('./Data/Videos/video11.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))
    
scene_list = detect('./Data/Queries/video11_1.mp4', ContentDetector())
for i, scene in enumerate(scene_list):
    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))
    
#  Scene 124: Start 00:04:35.500 / Frame 8265, End 00:04:49.433 / Frame 8683
#  Scene  2: Start 00:00:04.467 / Frame 134, End 00:00:18.400 / Frame 552


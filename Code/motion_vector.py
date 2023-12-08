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
    # Catch the error when there is no feature to track
    try:
        points2, st, err = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, points1, None)
        # Calculate motion vectors
        motion_vector = points2 - points1
    except:
        motion_vector = None

    # Calculate the average motion vector
    return motion_vector
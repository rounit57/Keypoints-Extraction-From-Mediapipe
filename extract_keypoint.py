
import numpy as np
import mediapipe as mp
import os
import cv2


mp_holistic = mp.solutions.holistic
features_old = np.zeros((468,3))

# function for keypoint detection
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False                 
    results = model.process(frame)                 
    frame.flags.writeable = True                  
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    return frame, results


# function to extract keypoints
def extract_keypoints(results):
    face=np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))
    pose=np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    lh=np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    rh=np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    return np.concatenate([pose, face, lh, rh])
    # return True,face,pose,lh,rh

video_path =''
save_video_path = ''


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    cap = cv2.VideoCapture(video_path) # read video file with cv2
    frame_count = 0
    while cap.isOpened():
        _ , frame = cap.read()
        if _: # check if frame received
            frame_count+=1
            frame, results = mediapipe_detection(frame, holistic)

            con_keypoints = extract_keypoints(results)
            
            cv2.putText(frame, f"Extracting features for '{video_path}'", (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            # if play_video:
            #     cv2.imshow('vid', frame)

            npy_path = os.path.join(save_video_path,str(frame_count))
            print(npy_path,len(con_keypoints))
            np.save(npy_path, con_keypoints)
        if not _: # if video ends, break
            break
        if cv2.waitKey(10) & 0xFF == ord('q'): # break if 'q' is pressed
            break
    cap.release()
    cv2.destroyAllWindows()

import cv2
import os
from mtcnn import MTCNN

video_folder = "dataset"
output_folder = "frames"

MAX_VIDEOS = 500
MAX_FRAMES = 30
FRAME_GAP = 10

detector = MTCNN()

def extract_frames(video_path, label):

    save_path = os.path.join(output_folder, label)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return

    video_name = os.path.basename(video_path).split('.')[0]

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_GAP == 0:
            if frame is None or frame.size ==0:
             continue
            try:
                faces = detector.detect_faces(frame)
            except:
                continue
            if faces:
                x, y, w, h = faces[0]['box']
                x=max(0, x)
                y=max(0, y)
                x2=min(frame.shape[1],x+w)
                y2=min(frame.shape[0],y+h)
                face = frame[y:y2, x:x2]
                if face.shape[0]>20 and face.shape[1]>20:

                    try:
                        face = cv2.resize(face, (128, 128))
                        frame_path = os.path.join(save_path, f"{video_name}_{saved}.jpg")
                        cv2.imwrite(frame_path, face)
                        saved += 1
                    except:
                        pass
            else:
                try:
                    frame_resized=cv2.resize(frame,(128,128))    
                    frame_path=os.path.join(save_path,f"{video_name}_{saved}.jpg")
                    cv2.imwrite(frame_path,frame_resized)   
                    saved += 1
                except:
                    pass
        if saved >= MAX_FRAMES:
            break

        count += 1

    cap.release()
    print(f"✅ {video_name} → {saved} faces")


for label in ["real", "fake"]:
    folder = os.path.join(video_folder, label)

    count_videos = 0

    for root, dirs, files in os.walk(folder):
        for video in files:
            if video.endswith(".mp4"):

                video_path = os.path.join(root, video)
                extract_frames(video_path, label)

                count_videos += 1
                if count_videos >= MAX_VIDEOS:
                    break
        if count_videos >= MAX_VIDEOS:
            break
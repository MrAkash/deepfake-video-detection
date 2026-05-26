import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

model = load_model("best_model.keras")
detector = MTCNN()

video_path = "C:\\Users\\Dell\\Pictures\\video_20260422_183145.mp4"
cap = cv2.VideoCapture(video_path)

predictions = []
frame_skip=10
max_prediction=30
frame_count=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count +=1
    if frame_count % frame_skip !=0:
        continue
    frame=cv2.resize(frame,(640,480))
    try:
     faces = detector.detect_faces(frame)
    except:
        continue
    if faces:
        x, y, w, h = faces[0]['box']

        # Fix bounding box
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        face = frame[y:y2, x:x2]

        # Check valid face
        if face.shape[0] > 20 and face.shape[1] > 20:
            try:
                face = cv2.resize(face, (128, 128))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face, verbose=0)[0][0]
                predictions.append(pred)
            except:
                pass
    if len(predictions)>=max_prediction:
        break
cap.release()

# Final decision
if len(predictions) == 0:
    print("❌ No faces detected in video")
else:
    avg_pred = np.mean(predictions)

    print("\n🔍 Prediction Score:", round(avg_pred, 3))
    print("frames Used", len(predictions))
    if avg_pred > 0.3:
        print("❌ FAKE VIDEO")
    else:
        print("✅ REAL VIDEO")


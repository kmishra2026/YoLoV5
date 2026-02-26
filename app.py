import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# ===============================
# Load YOLOv8 model (LIGHT VERSION)
# ===============================
MODEL_PATH = "yolov8n.pt"  # nano model is best for Render

model = YOLO(MODEL_PATH)
names = model.model.names

# ===============================
# Routes
# ===============================

@app.route('/')
def index():
    return render_template('index.html')


# ===============================
# Upload Video
# ===============================

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    os.makedirs('uploads', exist_ok=True)

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    return redirect(url_for('play_video', filename=file.filename))


@app.route('/uploads/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)


@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)


# ===============================
# Video Detection
# ===============================

def detect_objects_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))

        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f'{track_id} - {label}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1
                )

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )


@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(
        detect_objects_from_video(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ===============================
# Render PORT handling (VERY IMPORTANT)
# ===============================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

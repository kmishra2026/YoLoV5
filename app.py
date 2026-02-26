import os
import cv2
from flask import Flask, Response, request, redirect, url_for, send_from_directory, render_template_string
from ultralytics import YOLO

# Fix Ultralytics config warning on Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

app = Flask(__name__)

# Load lightweight model
model = YOLO("yolov8n.pt")
names = model.model.names

# =======================
# HTML TEMPLATES INLINE
# =======================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Object Detection</title>
</head>
<body style="text-align:center;margin-top:50px;">
    <h1>Object Detection</h1>

    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="video/*" required>
        <input type="submit" value="Upload Video">
    </form>

    <br><br>
    <a href="/start_webcam">Start Webcam Detection</a>
</body>
</html>
"""

VIDEO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Playback</title>
</head>
<body style="text-align:center;background:#f0f0f0;">
    <h1>Video Playback with Object Detection</h1>
    <img src="{{ url_for('video_feed', filename=filename) }}" width="1020" height="600">
    <br><br>
    <a href="/">Back to Home</a>
</body>
</html>
"""

WEBCAM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Webcam Feed</title>
</head>
<body style="text-align:center;background:#f0f0f0;">
    <h1>Webcam Object Detection</h1>
    <img src="{{ url_for('webcam_feed') }}" width="1020" height="600">
    <br><br>
    <a href="/">Back to Home</a>
</body>
</html>
"""

# =======================
# ROUTES
# =======================

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/start_webcam')
def start_webcam():
    return render_template_string(WEBCAM_HTML)

# =======================
# WEBCAM (⚠️ will NOT work on Render)
# =======================

def detect_objects_from_webcam():
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{track_id}-{label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =======================
# VIDEO UPLOAD
# =======================

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', file.filename)
    file.save(path)

    return render_template_string(VIDEO_HTML, filename=file.filename)

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{track_id}-{label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed/<filename>')
def video_feed(filename):
    path = os.path.join('uploads', filename)
    return Response(detect_objects_from_video(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =======================
# MAIN
# =======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
                

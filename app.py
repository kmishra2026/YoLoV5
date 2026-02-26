import os
import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

# Fix Ultralytics config on Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

app = Flask(__name__)

# Load lightweight YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.model.names

# =========================
# HTML (INLINE — no templates)
# =========================

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Object Detection</title>
</head>
<body style="text-align:center;margin-top:50px;background:#f4f4f4;">
    <h1>AI Image Object Detection</h1>

    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <br><br>
        <input type="submit" value="Detect Objects" style="padding:10px 20px;">
    </form>

    {% if image_url %}
        <h2>Detection Result</h2>
        <img src="{{ image_url }}" width="640">
    {% endif %}
</body>
</html>
"""

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template_string(HOME_HTML, image_url=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HOME_HTML, image_url=None)

    file = request.files["file"]
    if file.filename == "":
        return render_template_string(HOME_HTML, image_url=None)

    # Create folders
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)

    # Save uploaded image
    upload_path = os.path.join("static/uploads", file.filename)
    file.save(upload_path)

    # Read image
    img = cv2.imread(upload_path)

    # Run YOLO
    results = model(img)

    # Draw boxes
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls)]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )

    # Save result image
    result_path = os.path.join("static/results", file.filename)
    cv2.imwrite(result_path, img)

    image_url = "/" + result_path

    return render_template_string(HOME_HTML, image_url=image_url)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

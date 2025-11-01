import os
import uuid
import shutil
import redis
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Ensure necessary folders exist
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = os.path.join("static", "videos", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def home_page():
    processed_video_url = None

    if request.method == 'POST':
        uploaded_video = request.files.get("video")
        if uploaded_video:
            # Generate a unique job ID
            video_key = str(uuid.uuid4())
            temp_filename = os.path.join(UPLOAD_FOLDER, f"{video_key}.mp4")
            uploaded_video.save(temp_filename)

            # Process video with YOLO
            processed_video_url = process_video(temp_filename, video_key)

            # Remove temp upload
            os.remove(temp_filename)

    return render_template("home.html", processed_video=processed_video_url)


def process_video(temp_filename, video_key):
    """
    Runs YOLO detection and returns a Flask-accessible URL
    """
    # Run YOLO detection and save output
    results = model(temp_filename, save=True, conf=0.4)

    # Get the actual saved YOLO-processed file path
    save_dir = results[0].save_dir
    yolo_output_path = os.path.join(save_dir, os.path.basename(temp_filename))

    # Move YOLO output to predictable folder for Flask
    final_path = os.path.join(PROCESSED_FOLDER, f"{video_key}.mp4")
    shutil.copy(yolo_output_path, final_path)

    # Return URL for frontend
    return f"/static/videos/processed/{video_key}.mp4"


if __name__ == "__main__":
    app.run(debug=True)

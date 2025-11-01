import os
import uuid
import shutil
import redis
from flask import Flask, render_template, request, flash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# === Folders ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "videos", "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# === Redis Connection ===
r = redis.Redis(host="localhost", port=6379, db=0)

# === Load YOLO Model Once ===
model = YOLO("yolov8n.pt")


@app.route("/", methods=["GET", "POST"])
def home_page():
    processed_video_url = None

    if request.method == "POST":
        uploaded_video = request.files.get("video")

        if not uploaded_video:
            flash("No video file uploaded!", "error")
            return render_template("home.html", processed_video=None)

        try:
            # Generate unique key for the upload
            video_key = str(uuid.uuid4())
            temp_filename = os.path.join(UPLOAD_FOLDER, f"{video_key}.mp4")
            uploaded_video.save(temp_filename)

            # Process video
            processed_video_url = process_video(temp_filename, video_key)

        except Exception as e:
            flash(f"Error processing video: {e}", "error")
            processed_video_url = None

        finally:
            # Always clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    return render_template("home.html", processed_video=processed_video_url)


def process_video(temp_filename, video_key):
    """
    Runs YOLO detection and returns a URL to the processed video
    """
    print(f"[INFO] Running YOLO on {temp_filename}...")

    # Run YOLO detection and save output
    results = model(source=temp_filename, save=True, conf=0.4)

    # YOLO saves in something like runs/detect/predict*
    save_dir = results[0].save_dir
    predicted_name = os.path.basename(temp_filename)
    yolo_output_path = os.path.join(save_dir, predicted_name)

    # Ensure the YOLO output file exists
    if not os.path.exists(yolo_output_path):
        raise FileNotFoundError(f"Processed video not found at {yolo_output_path}")

    # Move YOLO output to Flask static folder
    final_path = os.path.join(PROCESSED_FOLDER, f"{video_key}.mp4")
    shutil.move(yolo_output_path, final_path)

    # Return the URL Flask can serve
    return f"/static/videos/processed/{video_key}.mp4"


if __name__ == "__main__":
    app.run(debug=True)

import os
import uuid
import shutil
import redis
from flask import Flask, render_template, request, flash
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "supersecretkey"  


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "static", "videos", "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

r = redis.Redis(host="localhost", port=6379, db=0)

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
    # Run YOLO detection and save output
    results = model(source=temp_filename, save=True, conf=0.4)
    print("\n\nSTEP 1. YOLO loaded\n\n")

    # Get the actual YOLO output path (may be .avi)
    yolo_output_path = str(results[0].path)
    print(f"\n\nSTEP 2. YOLO saved at {yolo_output_path}\n\n")

    # Ensure the YOLO output file exists
    if not os.path.exists(yolo_output_path):
        print("\n\nSTEP 3. File not found!\n\n")
        raise FileNotFoundError(f"Processed video not found at {yolo_output_path}")

    print("\n\nSTEP 3. Found YOLO output\n\n")

    # Move YOLO output to Flask static folder and keep the .avi extension
    file_extension = os.path.splitext(yolo_output_path)[1]  # e.g., '.avi'
    final_path = os.path.join(PROCESSED_FOLDER, f"{video_key}{file_extension}")

    print(f"\n\nSTEP 4. Moving to {final_path}\n\n")
    shutil.move(yolo_output_path, final_path)

    print("\n\nSTEP 5. Returning video URL\n\n")

    # Return the URL Flask can serve (still uses the original extension)
    return f"/static/videos/processed/{video_key}{file_extension}"



if __name__ == "__main__":
    app.run(debug=True)

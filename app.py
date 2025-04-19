import os
import cv2
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash
)

UPLOAD_FOLDER = 'static/uploads'
PROCESS_FOLDER = 'static/process'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Create app
app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESS_FOLDER'] = PROCESS_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESS_FOLDER, exist_ok=True)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('video')
        if file and file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Save uploaded video
            filename = file.filename
            base, ext = os.path.splitext(filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            # Open video capture
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Use MJPG codec and AVI container
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_filename = f'processed_{base}.avi'
            output_path     = os.path.join(app.config['PROCESS_FOLDER'], output_filename)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Could not open MJPG/AVI VideoWriter for '{output_path}'")

            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                out.write(frame)

            cap.release()
            out.release()

            flash("✅ Video processed successfully!", "success")
            return render_template('index.html',
                                   processed_video=url_for('process_file',
                                                           filename=output_filename))
        else:
            flash("❌ Please upload a valid video file (.mp4, .avi, .mov)", "error")
            return redirect(request.url)

    return render_template('index.html')

@app.route('/process_file/<filename>')
def process_file(filename):
    return send_from_directory(app.config['PROCESS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

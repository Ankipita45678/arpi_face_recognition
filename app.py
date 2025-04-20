from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import base64
import os

app = Flask(__name__)

# --- Step 1: Load known faces ---
known_face_encodings = []
known_face_filenames = []
known_faces_dir = "known_faces"

for file in os.listdir(known_faces_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_faces_dir, file)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_filenames.append(file)  # Keep original filename

# --- Step 2: Load real names from names.txt ---
filename_to_realname = {}

with open("known_faces/names.txt", "r") as f:
    for line in f:
        if ':' in line:
            file, realname = line.strip().split(':', 1)
            filename_to_realname[file.strip()] = realname.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image'}), 400

    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    name = "Unknown"
    greeting = "No known person detected."

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            best_match_index = matches.index(True)
            matched_filename = known_face_filenames[best_match_index]
            name = filename_to_realname.get(matched_filename, matched_filename.split('.')[0])
            greeting = f"Hi {name}!"
            break

    return jsonify({'name': name, 'greeting': greeting})

if __name__ == '__main__':
    app.run(debug=True)

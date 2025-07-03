from flask import Flask, render_template, redirect, url_for, session, request, Response, jsonify
import random
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define the path to the static folder for local video files
video_folder = os.path.join('static', 'videos')

videos = [
    # Happy Videos
    {"id": 1, "title": "Funny Dog Video", "tags": ["happy"], "src": "videos/- I have crossed over to the land of dreams DOG.mp4"},
    {"id": 2, "title": "Redbull Skydiving", "tags": ["happy", "surprise"], "src": "videos/Having fun after handing over some energy Redbull.mp4"},
    {"id": 3, "title": "Happy Dog", "tags": ["happy"], "src": "videos/Happy dog..mp4"},
    {"id": 4, "title": "Petting Cat", "tags": ["happy"], "src": "videos/PETHIM!.mp4"},
    {"id": 5, "title": "Cat Videos", "tags": ["happy"], "src": "videos/Cute Cat Video.mp4"},
    {"id": 6, "title": "MotoGP Stunt", "tags": ["happy"], "src": "videos/MotoGP Stunt.mp4"},
    {"id": 7, "title": "Best Funny Moments", "tags": ["happy"], "src": "https://www.youtube.com/embed/pIvf9bOPXIw"},
    {"id": 8, "title": "Laugh Out Loud", "tags": ["happy"], "src": "https://www.youtube.com/embed/OcmcptbsvzQ"},

    # Surprise Videos
    {"id": 9, "title": "Redbull F1 stunt", "tags": ["surprise"], "src": "videos/I like it.. Let's add a stunt.mp4"},
    {"id": 10, "title": "Bike Stunt", "tags": ["surprise"], "src": "videos/Red Bull X-Fighters Madrid 2014 edit.mp4"},
    {"id": 11, "title": "Skydive game", "tags": ["surprise"], "src": "videos/Skydive catch.mp4"},
    {"id": 12, "title": "Redbull plane", "tags": ["surprise"], "src": "videos/Redbull airplane.mp4"},
    {"id": 13, "title": "Amazing Stunts", "tags": ["surprise"], "src": "https://www.youtube.com/embed/VWw_1-gEdLA"},
    {"id": 14, "title": "Incredible Moments", "tags": ["surprise"], "src": "https://www.youtube.com/embed/IZHwZwfy9W0"},
    {"id": 15, "title": "Unexpected Events", "tags": ["surprise"], "src": "https://www.youtube.com/embed/CFWYi6gOf_Q"},
    {"id": 16, "title": "Surprising Tricks", "tags": ["surprise"], "src": "https://www.youtube.com/embed/rXi2sjYcUgU"},

    # Anger Videos
    {"id": 17, "title": "Anger Management", "tags": ["angry"], "src": "https://www.youtube.com/embed/kmTEyxWg7Hs"},
    {"id": 18, "title": "Dealing with Anger", "tags": ["angry"], "src": "https://www.youtube.com/embed/8X0gNEbFWKw"},
    {"id": 19, "title": "Calm Your Mind", "tags": ["angry"], "src": "https://www.youtube.com/embed/a8t5SwmqDnk"},
    {"id": 20, "title": "Meditation for Anger", "tags": ["angry"], "src": "https://www.youtube.com/embed/xLWtmaAV5Mo"},
    {"id": 21, "title": "Peace of Mind", "tags": ["angry"], "src": "https://www.youtube.com/embed/JV61dx977dU"},

    # Neutral Videos
    {"id": 22, "title": "Daily Routine", "tags": ["neutral"], "src": "https://www.youtube.com/embed/BRXdjc5yxVs"},
    {"id": 23, "title": "Life Skills", "tags": ["neutral"], "src": "https://www.youtube.com/embed/FkQWpQd9Zdo"},
    {"id": 24, "title": "How-To Guide", "tags": ["neutral"], "src": "https://www.youtube.com/embed/DFHuu6yZiH8"},
    {"id": 25, "title": "Educational Content", "tags": ["neutral"], "src": "https://www.youtube.com/embed/ZVO8Wt_PCgE"},
    {"id": 26, "title": "Learning New Things", "tags": ["neutral"], "src": "https://www.youtube.com/embed/48vET0WDkqc"},
    {"id": 27, "title": "Daily Tips", "tags": ["neutral"], "src": "https://www.youtube.com/embed/a3jeI-kWrO4"},

    # Sad Videos
    {"id": 28, "title": "Emotional Journey", "tags": ["sad"], "src": "https://www.youtube.com/embed/YDqGcmHHOys"},
    {"id": 29, "title": "Moving Forward", "tags": ["sad"], "src": "https://www.youtube.com/embed/d96akWDnx0w"},
    {"id": 30, "title": "Hope and Healing", "tags": ["sad"], "src": "https://www.youtube.com/embed/nAnrHFIRPh0"},
    {"id": 31, "title": "Finding Strength", "tags": ["sad"], "src": "https://www.youtube.com/embed/gJEp-JoQVIw"},
    {"id": 32, "title": "Inspirational Stories", "tags": ["sad"], "src": "https://www.youtube.com/embed/1kIFrf5OPxE"},
    {"id": 33, "title": "Healing Process", "tags": ["sad"], "src": "https://www.youtube.com/embed/jYn1ZlvfPHU"},
    {"id": 34, "title": "Journey to Recovery", "tags": ["sad"], "src": "https://www.youtube.com/embed/I09YxDXODGg"},

    # Fear Videos
    {"id": 35, "title": "Overcoming Fear", "tags": ["fear"], "src": "https://www.youtube.com/embed/gJEp-JoQVIw"},
    {"id": 36, "title": "Face Your Fears", "tags": ["fear"], "src": "https://www.youtube.com/embed/cgMvFRUAd0s"},
    {"id": 37, "title": "Courage Building", "tags": ["fear"], "src": "https://www.youtube.com/embed/8plwv25NYRo"},
    {"id": 38, "title": "Brave Moments", "tags": ["fear"], "src": "https://www.youtube.com/embed/eAK14VoY7C0"},
    {"id": 39, "title": "Overcoming Challenges", "tags": ["fear"], "src": "https://www.youtube.com/embed/79kpoGF8KWU"},

    # Disgust Videos
    {"id": 40, "title": "Clean Living", "tags": ["disgust"], "src": "https://www.youtube.com/embed/linlz7-Pnvw"},
    {"id": 41, "title": "Better Choices", "tags": ["disgust"], "src": "https://www.youtube.com/embed/nW-IwT65A6U"},
    {"id": 42, "title": "Healthy Living", "tags": ["disgust"], "src": "https://www.youtube.com/embed/ImTqvWxc2Fo"},
    {"id": 43, "title": "Lifestyle Changes", "tags": ["disgust"], "src": "https://www.youtube.com/embed/obbk6TL0nhE"},
    {"id": 44, "title": "Fresh Start", "tags": ["disgust"], "src": "https://www.youtube.com/embed/9lVB1-c69Sw"},
    {"id": 45, "title": "New Beginnings", "tags": ["disgust"], "src": "https://www.youtube.com/embed/_ktvAxDMj5k"}
]

# Load the emotion recognition model
model_json_file = 'facialemotionmodel.json'
model_weights_file = 'facialemotionmodel.h5'

with open(model_json_file, 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.load_weights(model_weights_file)

# Load the face cascade
face_cascade = cv2.CascadeClassifier('haar_cascade.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion')
def detect_emotion():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                
                predictions = model.predict(roi_gray)
                max_index = np.argmax(predictions[0])
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                predicted_emotion = emotion_labels[max_index]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_emotion')
def get_emotion():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        return jsonify({"error": "Failed to capture image"}), 400

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    x, y, w, h = faces[0]
    roi_gray = gray_frame[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_gray = roi_gray.astype('float32') / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = np.expand_dims(roi_gray, axis=-1)
    
    predictions = model.predict(roi_gray)
    max_index = np.argmax(predictions[0])
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[max_index].lower()

    return jsonify({"emotion": predicted_emotion})

@app.route('/recommend_video')
def recommend_video():
    emotion = request.args.get('emotion', 'happy').lower()
    filtered_videos = [video for video in videos if emotion in video['tags']]
    if filtered_videos:
        recommended_video = random.choice(filtered_videos)
        session['filtered_videos'] = filtered_videos
        session['current_index'] = 0
        return render_template('video.html', video=recommended_video, current_index=0, total=len(filtered_videos))
    else:
        return render_template('no_videos.html', emotion=emotion)

@app.route('/video')
def video():
    current_index = session.get('current_index', 0)
    filtered_videos = session.get('filtered_videos', videos)

    if not filtered_videos:
        return render_template('no_videos.html', emotion="Unknown")

    video = filtered_videos[current_index]
    total = len(filtered_videos)
    return render_template('video.html', video=video, current_index=current_index, total=total)

@app.route('/next')
def next_video():
    filtered_videos = session.get('filtered_videos', videos)
    current_index = session.get('current_index', 0)
    if current_index < len(filtered_videos) - 1:
        current_index += 1
        session['current_index'] = current_index
    return redirect(url_for('video'))

@app.route('/previous')
def previous_video():
    filtered_videos = session.get('filtered_videos', videos)
    current_index = session.get('current_index', 0)
    if current_index > 0:
        current_index -= 1
        session['current_index'] = current_index
    return redirect(url_for('video'))

if __name__ == '__main__':
    app.run(debug=True)
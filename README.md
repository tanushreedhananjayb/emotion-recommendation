#🎭 Emotion-Based Video Recommendation System

This project is a web-based application that detects a user's facial emotion in real-time and recommends videos based on the detected emotion using computer vision and deep learning.

---

🧠 Features

- 📷 **Real-time emotion detection** using OpenCV and Haar Cascades.
- 🤖 **Trained CNN model** to recognize emotions from facial features.
- 🎬 **Emotion-based content recommendation** (videos change based on detected emotion).
- 🌐 **Flask-based web interface** for seamless interaction.
- 🗂️ Organized with templates, static files, and modular Python code.

---

 😄 Supported Emotions

- Happy 😊
- Sad 😢
- Angry 😠
- Surprised 😲
- Neutral 😐
- Fear 😨
- Disgust 🤢

---
🛠️ Tech Stack

| Layer          | Tools Used                             |
|----------------|------------------------------------------|
| Frontend       | HTML, CSS, Bootstrap (via Flask Jinja templates) |
| Backend        | Python, Flask                          |
| ML/DL          | Keras, TensorFlow, OpenCV, NumPy       |
| Deployment     | Localhost (can be deployed to Heroku/AWS) |

---
 📁 Project Structure

emotion-recommendation/
│
├── app.py # Main Flask app
├── realtimedetection.py # Standalone real-time emotion detection script
├── facialemotionmodel.h5 # Trained CNN model
├── facialemotionmodel.json # Model architecture
├── haar_cascade.xml # Haar cascade classifier
├── requirements.txt
├── static/
│ └── videos/ # Video recommendations per emotion
├── templates/
│ ├── index.html # Landing page
│ └── video.html # Video recommendation page
└── .gitignore


---

🖥️ How to Run

> ⚠️ Ensure Python 3.10+ and `virtualenv` are installed.

1. Clone the Repository:

```bash
git clone https://github.com/tanushreedhananjayb/emotion-recommendation.git
cd emotion-recommendation

2. Create and Activate a Virtual Environment:

python -m venv venv
venv\Scripts\activate        # For Windows
# OR
source venv/bin/activate     # For Mac/Linux

3. Install Requirements:

pip install -r requirements.txt

4. Run the App:

python app.py

Then open your browser at: http://127.0.0.1:5000/

🎥 Sample Videos
Video recommendations for each emotion are stored in /static/videos/. These videos play depending on the user's detected emotion in real-time.

👩‍💻 Author
Tanushree Dhananjay Bhamare
B.Tech in CSBS, RAIT 



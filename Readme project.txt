(Before running: installing flask and creating and activating virtual environment(venv))
in VSCode terminal
	pip install flask
	python -m venv venv
	venv\Scripts\activate
in powershell(if venv doesn't work)
	Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser (YES TO ALL)
	.\venv\Scripts\Activate


Folder
video-recommendation-system/
│
├── app.py                 # Main Flask app
├── templates/
│   └── video.html         # HTML template for displaying the video
├── venv/                  # Virtual environment directory (created by `venv`)
└── Readme project.txt     

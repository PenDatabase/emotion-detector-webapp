# Emotion Detection Web Application

AI-powered web application that detects human emotions from images or live webcam capture.

## 🎯 Features

- **Upload Image**: Detect emotions from uploaded images
- **Webcam Capture**: Real-time emotion detection from webcam
- **7 Emotions**: Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust
- **Database Storage**: Stores all detections with user names and timestamps
- **History View**: View past detection results
- **Statistics**: Emotion distribution analytics

## 📁 Project Structure

```
SURNAME_MATNO_EMOTION_DETECTION_WEB_APP/
├── app.py                          # Flask backend
├── model.py                        # Model training script
├── emotion_guardian_model.h5       # Trained model (generated)
├── emotion_detection.db            # SQLite database (generated)
├── requirements.txt                # Python dependencies
├── link_to_my_web_app.txt         # Hosting link
├── README.md                       # This file
├── templates/
│   └── index.html                 # Web interface
└── static/
    ├── style.css                  # Styling
    └── uploads/                   # Uploaded images (generated)
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python model.py
```

**Options:**
- Use FER2013 dataset (recommended for real training)
- Use sample data (for quick demonstration)

The trained model will be saved as `emotion_guardian_model.h5`

### 3. Run the Web Application

```bash
python app.py
```

Open your browser and navigate to: `http://127.0.0.1:5000`

## 📊 Database Schema

SQLite database (`emotion_detection.db`) with table:

```sql
detections (
    id INTEGER PRIMARY KEY,
    user_name TEXT,
    image_path TEXT,
    emotion_result TEXT,
    confidence REAL,
    detection_type TEXT,
    timestamp DATETIME
)
```

## 🌐 Deployment

### Recommended Free Hosting Platforms:

1. **Render** (https://render.com)
   - Easy deployment
   - Free tier available
   - Good for Python apps

2. **Railway** (https://railway.app)
   - Simple setup
   - Free credits

3. **PythonAnywhere** (https://www.pythonanywhere.com)
   - Python-specific hosting
   - Free tier

### Deployment Steps (Render Example):

1. Create a GitHub repository
2. Push your code to GitHub
3. Sign up on Render
4. Create new Web Service
5. Connect your GitHub repository
6. Set build command: `pip install -r requirements.txt`
7. Set start command: `gunicorn app:app`
8. Deploy!

## 🎓 Assignment Submission

1. ✅ Upload project to GitHub repository
2. ✅ Deploy to free hosting platform
3. ✅ Update `link_to_my_web_app.txt` with hosting URL
4. ✅ Zip the entire project folder
5. ✅ Submit to: odunayo.osofuye@covenantuniversity.edu.ng

## 📝 Notes

- Model file (`emotion_guardian_model.h5`) must be present to run the app
- Database is created automatically on first run
- Upload folder is created automatically
- For production, consider using a proper database (PostgreSQL)

## 👨‍💻 Author

**Your Name** - Covenant University

## 📜 License

Academic Project - Covenant University Assignment
```
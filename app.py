"""
Emotion Detection Web Application - Backend
Flask server for handling image uploads, webcam captures, and emotion detection
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from datetime import datetime
import base64
from tensorflow import keras
import sqlite3
import json
import logging
import sys
import traceback

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder created/verified: {app.config['UPLOAD_FOLDER']}")

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained model
MODEL_PATH = 'emotion_guardian_model.h5'
try:
    model = keras.models.load_model(MODEL_PATH)
    logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"✗ Error loading model: {e}")
    logger.error(traceback.format_exc())
    model = None

# Load face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logger.info("Face cascade classifier loaded")


# Database Setup
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            emotion_result TEXT NOT NULL,
            confidence REAL,
            detection_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized")


def save_to_database(user_name, image_path, emotion_result, confidence, detection_type):
    """Save detection result to database"""
    try:
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (user_name, image_path, emotion_result, confidence, detection_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_name, image_path, emotion_result, confidence, detection_type))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_face(face_img):
    """Preprocess face image for model prediction"""
    # Resize to model input size
    face_img = cv2.resize(face_img, (48, 48))
    
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    face_img = face_img / 255.0
    
    # Reshape for model input
    face_img = face_img.reshape(1, 48, 48, 1)
    
    return face_img


def detect_emotion(image_path):
    """Detect emotion from image file"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None, "No face detected"
    
    # Get the first face
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Preprocess and predict
    processed_face = preprocess_face(face_roi)
    predictions = model.predict(processed_face, verbose=0)
    
    # Get emotion with highest probability
    emotion_idx = np.argmax(predictions[0])
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = float(predictions[0][emotion_idx]) * 100
    
    # Draw rectangle and label on image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f"{emotion} ({confidence:.1f}%)", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save annotated image
    annotated_path = image_path.replace('.', '_annotated.')
    cv2.imwrite(annotated_path, img)
    
    return emotion, confidence, annotated_path


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and emotion detection"""
    try:
        logger.info("Upload request received")
        
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Check if file is in request
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        user_name = request.form.get('user_name', 'Anonymous')
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Detect emotion
            emotion, confidence, annotated_path = detect_emotion(filepath)
            
            if emotion is None:
                logger.warning("No face detected in uploaded image")
                return jsonify({
                    'error': 'Could not detect face in image',
                    'image_url': url_for('static', filename=f'uploads/{filename}')
                }), 400
            
            # Save to database
            save_to_database(user_name, filepath, emotion, confidence, 'upload')
            logger.info(f"Emotion detected: {emotion} ({confidence}%)")
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': round(confidence, 2),
                'image_url': url_for('static', filename=f'uploads/{os.path.basename(annotated_path)}'),
                'user_name': user_name
            })
        
        logger.warning("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/webcam', methods=['POST'])
def webcam_capture():
    """Handle webcam capture and emotion detection"""
    try:
        logger.info("Webcam request received")
        
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get base64 image from request
        data = request.get_json()
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        image_data = data.get('image')
        user_name = data.get('user_name', 'Anonymous')
        
        if not image_data:
            logger.error("No image data in request")
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Save image
        filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
        logger.info(f"Webcam image saved to: {filepath}")
        
        # Detect emotion
        emotion, confidence, annotated_path = detect_emotion(filepath)
        
        if emotion is None:
            logger.warning("No face detected in webcam capture")
            return jsonify({'error': 'Could not detect face'}), 400
        
        # Save to database
        save_to_database(user_name, filepath, emotion, confidence, 'webcam')
        logger.info(f"Webcam emotion detected: {emotion} ({confidence}%)")
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'image_url': url_for('static', filename=f'uploads/{os.path.basename(annotated_path)}'),
            'user_name': user_name
        })
        
    except Exception as e:
        logger.error(f"Error in webcam_capture: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/history')
def history():
    """View detection history from database"""
    try:
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_name, emotion_result, confidence, detection_type, timestamp
            FROM detections
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        history_data = []
        for row in results:
            history_data.append({
                'user_name': row[0],
                'emotion': row[1],
                'confidence': row[2],
                'type': row[3],
                'timestamp': row[4]
            })
        
        return jsonify(history_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get statistics about detections"""
    try:
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM detections')
        total = cursor.fetchone()[0]
        
        # Emotion distribution
        cursor.execute('''
            SELECT emotion_result, COUNT(*) as count
            FROM detections
            GROUP BY emotion_result
            ORDER BY count DESC
        ''')
        
        emotion_dist = dict(cursor.fetchall())
        
        conn.close()
        
        return jsonify({
            'total_detections': total,
            'emotion_distribution': emotion_dist
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("EMOTION DETECTION WEB APP")
    print("=" * 60)
    
    # Initialize database
    init_database()
    
    # Run Flask app
    print("\n✓ Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        print("✓ Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
else:
    # Initialize database when running with gunicorn
    init_database()
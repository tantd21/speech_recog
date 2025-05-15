from flask import Flask, request, render_template, jsonify
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import os
import scipy.io.wavfile as wav
import io
from pydub import AudioSegment
import tempfile
import subprocess
import shutil

app = Flask(__name__)

# FFmpeg path - Update this path to your ffmpeg location
FFMPEG_PATH = r"C:\Users\thaid\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# Check if ffmpeg is installed
def check_ffmpeg():
    try:
        if os.path.exists(FFMPEG_PATH):
            subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, check=True)
            return True
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Load model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    print(f"Warning: Error loading model: {str(e)}")
    print("Please place your model file in the root directory.")

def convert_audio_to_wav(input_path, output_path=None):
    """Convert any audio format to wav format"""
    try:
        if not check_ffmpeg():
            raise RuntimeError(f"ffmpeg not found at {FFMPEG_PATH}. Please check the path.")
        
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + '.wav'
            
        # Use ffmpeg to convert audio
        subprocess.run([
            FFMPEG_PATH, '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '22050',
            '-ac', '1',
            output_path
        ], capture_output=True, check=True)
        
        return output_path
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return None

def extract_features(file_path):
    """
    Extract features from audio file
    """
    try:
        # Read audio file
        audio_data, sample_rate = librosa.load(file_path, sr=22050, mono=True)
        
        # Ensure audio is not empty
        if len(audio_data) == 0:
            raise ValueError("Audio file is empty")
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        
        # Normalize MFCCs
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Ensure we have the right shape
        if len(mfccs_scaled) != 40:
            raise ValueError(f"Expected 40 MFCC features, got {len(mfccs_scaled)}")
            
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/main')
def main():
    if not check_ffmpeg():
        return render_template('index.html', error=f"ffmpeg not found at {FFMPEG_PATH}. Please check the path.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    if not check_ffmpeg():
        return jsonify({'error': f'ffmpeg not found at {FFMPEG_PATH}. Please check the path.'}), 500
    
    temp_input = None
    temp_wav = None
    
    try:
        # Save the uploaded file temporarily
        file = request.files['audio']
        
        # Get file extension
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Create temporary input file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_input.write(file.read())
        temp_input.close()
        
        # Convert to wav if needed
        if file_ext != '.wav':
            temp_wav = convert_audio_to_wav(temp_input.name)
            if not temp_wav:
                raise ValueError("Failed to convert audio format")
            audio_path = temp_wav
        else:
            audio_path = temp_input.name
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            raise ValueError("Error processing audio features")
            
        # Prepare features for prediction
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        prediction = model.predict(features)
        emotion_idx = np.argmax(prediction[0])
        
        # Map emotion index to label
        # emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

        predicted_emotion = emotion_labels[emotion_idx]
        
        return jsonify({
            'emotion': predicted_emotion,
            'confidence': float(prediction[0][emotion_idx])
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temporary files
        if temp_input and os.path.exists(temp_input.name):
            os.remove(temp_input.name)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

if __name__ == '__main__':
    app.run(debug=True) 
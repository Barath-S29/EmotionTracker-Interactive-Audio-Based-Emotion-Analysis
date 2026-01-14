# 🎭 EmotionTracker: Interactive Audio-Based Emotion Analysis

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.3-green.svg)](https://flask.palletsprojects.com/)
A Flask web application that analyzes emotions from audio recordings using machine learning. Users fill out a survey form (text and radio buttons), then record a 3-second audio clip. The system extracts prosodic features and classifies emotions using a Support Vector Machine (SVM) model.

## Features

- Audio recording from microphone (3 seconds)
- Emotion classification using prosodic features (pitch and intensity)
- Five emotion classes: Happiness, Sad, Angry, Fear, Neutral
- Web interface with survey form
- Tkinter GUI popup for results display
- Optional email notifications
- Utility script to organize audio files by predicted emotions

## Technologies Used

- **Python 3.x** - Core language
- **Flask 2.2.3** - Web framework
- **scikit-learn 1.1.3** - SVM classifier
- **Parselmouth 0.4.0** - Speech analysis and feature extraction
- **PyAudio 0.2.11** - Audio I/O
- **NumPy 1.23.5** - Numerical computations
- **Tkinter** - GUI display (built-in)

## Features 
- **🎤 Real-time Audio Recording**: Capture 3-second audio clips directly from the microphone
- **🧠 Emotion Detection**: ML-powered emotion classification using prosodic features (pitch, intensity)
- **📊 Five Emotion Classes**: Detects Happiness, Sadness, Anger, Fear, and Neutral emotions
- **🌐 Web Interface**: Flask-based web application with survey form (text input and radio buttons)
- **🖥️ GUI Display**: Tkinter popup window showing predicted emotion with emojis
- **📧 Email Notifications**: Optional email alerts with emotion predictions and audio attachments
- **📁 Dataset Organization**: Utility script to organize audio files by predicted emotions
- **🔒 Secure Configuration**: Environment variables and config files for sensitive data


## 📁 Project Structure

```
EmotionTracker-Interactive-Audio-Based-Emotion-Analysis/
│
├── Emotion.py                 # Main Flask application
├── emotion_gui.py             # Tkinter GUI module
├── organize_archive.py        # Dataset organization utility
├── config.example.py         # Email configuration template
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── Dataset/                   # Audio dataset directory
│   ├── Angry/                # Angry emotion samples
│   ├── Fear/                 # Fear emotion samples
│   ├── Happiness/            # Happiness emotion samples
│   ├── Neutral/              # Neutral emotion samples
│   ├── Sad/                  # Sad emotion samples
│   └── archive/              # Archive folder (optional)
│
├── templates/                 # Flask HTML templates
│   ├── survey_form.html      # Survey form page
│   └── thank_you.html        # Thank you/results page
│
└── recordings/                # User recordings (created at runtime, gitignored)
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Microphone for audio recording

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/EmotionTracker-Interactive-Audio-Based-Emotion-Analysis.git
cd EmotionTracker-Interactive-Audio-Based-Emotion-Analysis
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: On some systems, PyAudio may require additional dependencies:
- **Linux**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Mac**: `brew install portaudio`

### Step 3: Prepare Dataset

Organize your audio files in the `Dataset/` folder:

```
Dataset/
├── Angry/
│   └── *.wav files
├── Fear/
│   └── *.wav files
├── Happiness/
│   └── *.wav files
├── Neutral/
│   └── *.wav files
└── Sad/
    └── *.wav files
```

Use `organize_archive.py` to automatically organize files from an archive folder.

## Configuration

### Email Configuration (Optional)

To enable email notifications, you have two options:
**Option 1: Using Config File**

1. Copy the example config:
   ```bash
   cp config.example.py config.py
   ```

**Note**: For Gmail, you need to create an [App Password](https://www.getmailbird.com/gmail-app-password/).
2. Edit `config.py` with your credentials:
   ```python
   SENDER_EMAIL = "your_email@gmail.com"
   SENDER_PASSWORD = "your_app_password"
   RECEIVER_EMAIL = "receiver@gmail.com"
   ```

**Option 2: Using Environment Variables**

```bash
# Windows (PowerShell)
$env:SENDER_EMAIL="your_email@gmail.com"
$env:SENDER_PASSWORD="your_app_password"
$env:RECEIVER_EMAIL="receiver@gmail.com"

# Linux/Mac
export SENDER_EMAIL="your_email@gmail.com"
export SENDER_PASSWORD="your_app_password"
export RECEIVER_EMAIL="receiver@gmail.com"
```

**Note**: For Gmail, use an [App Password](https://www.getmailbird.com/gmail-app-password/). The `config.py` file is gitignored.

## Usage

### Running the Application

1. Start the Flask server:
   ```bash
   python Emotion.py
   ```

2. The application will:
   - Load and train the SVM model on your dataset
   - Start the Flask web server on port 5000
   - Display training progress in the console

3. Open your browser:
   ```
   http://127.0.0.1:5000/survey
   ```

4. Use the application:
   - **Type** your name in the text field
   - **Click** Yes/No radio buttons to answer survey questions (no typing or audio needed for questions)
   - Click "Record" button to submit the form and start audio recording
   - **Speak** for 3 seconds (recording stops automatically) - this audio is used for emotion detection only

5. View results:
   - Tkinter GUI window shows predicted emotion
   - Web page displays thank you message with result
   - Email sent if configured

### Organizing Archive Files

To organize audio files from an archive folder:

```bash
python organize_archive.py
```

The script will:
1. Train a model on your existing dataset
2. Predict emotions for files in `Dataset/archive/`
3. Organize them into emotion folders
4. Show statistics

**Options**:
- **Copy mode** (default): Keeps original files in archive
- **Move mode**: Moves files from archive to emotion folders

## Machine Learning Model

### Algorithm

Uses a **Support Vector Machine (SVM)** with linear kernel for emotion classification.

### Feature Extraction

Prosodic features extracted using Parselmouth:
- Pitch mean and standard deviation
- Intensity mean and standard deviation

### Model Training

- Training split: 80% of dataset
- Test split: 20% of dataset
- Kernel: Linear
- Regularization: C=1.0
- Random state: 42

### Emotion Classes

- Happiness
- Sad
- Angry
- Fear
- Neutral

## Dataset

The dataset consists of WAV audio files organized by emotion. Current distribution:

- **Total Samples**: ~1,450 audio files
- **Format**: WAV files
- **Distribution**:
  - Happiness: ~510 files (35.2%)
  - Fear: ~400 files (27.6%)
  - Neutral: ~300 files (20.7%)
  - Angry: ~178 files (12.3%)
  - Sad: ~62 files (4.3%)

### Dataset Sources

- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Custom recordings
- Public emotion datasets
**Note**: The dataset is imbalanced, which may affect model performance. Consider data augmentation for underrepresented classes.

## API Endpoints

### GET `/survey`

Displays the survey form.

### POST `/survey`

Processes survey submission and records audio.

**Form Data**:
- `name`: User's name
- `q0`, `q1`, `q2`: Survey question responses

**Response**: 
- HTML thank you page with predicted emotion
- Tkinter GUI popup (separate thread)
- Email notification (if configured, separate thread)

## Troubleshooting

### "No audio files found in the dataset folder"

- Ensure `Dataset/` contains emotion subfolders (Angry, Fear, Happiness, Neutral, Sad)
- Check that each folder contains `.wav` files
- Verify dataset path in `Emotion.py` (line 238)

### PyAudio Installation Errors

**Windows**:
```bash
pip install pipwin
pipwin install pyaudio
```

**Linux**:
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Mac**:
```bash
brew install portaudio
pip install pyaudio
```

### "Model not trained" Error

- Check that dataset folders contain valid `.wav` files
- Verify Parselmouth is installed correctly
- Check console for error messages during training

### Microphone Not Working

- Check microphone permissions in system settings
- Verify microphone is connected
- Try running as administrator (Windows) or with sudo (Linux)

### Email Not Sending

- Verify email credentials in `config.py` or environment variables
- For Gmail, use an App Password (not regular password)
- Check SMTP server settings
- Review console for error messages

### Parselmouth Installation Issues

- Parselmouth requires Praat to be installed
- Download Praat from: https://www.fon.hum.uva.nl/praat/
- Ensure Praat is in your system PATH

## Acknowledgements

- Kaggle for datasets and inspiration
- RAVDESS dataset contributors
- Parselmouth, Flask, and scikit-learn developers

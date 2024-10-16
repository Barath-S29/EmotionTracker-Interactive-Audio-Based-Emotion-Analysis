# **EmotionTracker: Interactive Audio-Based Emotion Analysis**
EmotionTracker is an interactive platform designed to analyze and detect emotions from audio recordings. By integrating a user-friendly web survey with real-time voice capture, the system utilizes prosodic features to classify emotions using a Support Vector Machine (SVM) model. The application not only provides immediate feedback on users' emotional states but also facilitates communication through email notifications. With a focus on enhancing emotional awareness, EmotionTracker aims to support users in understanding their feelings in a seamless and engaging manner.

## **Overview**
EmotionTracker is an interactive platform that utilizes audio analysis to detect emotions from user recordings. The project combines a survey form with real-time voice recording and emotion recognition capabilities, providing users with immediate feedback on their emotional state.

## **Features**
- **Audio Recording:** Capture audio input directly from the microphone.
- **Emotion Detection:** Analyze audio to predict emotional states using prosodic features.
- **Email Notifications:** Send email summaries of results to users.
- **User Interface:** Simple web-based survey form with an interactive GUI to display results.
- **Data Handling:** Organizes audio recordings and emotion predictions efficiently.

## **Technologies Used**
- Python
- Flask (Web Framework)
- PyAudio (Audio Processing)
- Parselmouth (Speech Analysis)
- scikit-learn (Machine Learning)
- Tkinter (GUI)
- HTML/CSS (Frontend)

## **Machine Learning Algorithm Used**
EmotionTracker utilizes a **Support Vector Machine (SVM)** for emotion classification. The SVM is trained on prosodic features extracted from audio recordings, allowing it to effectively differentiate between various emotional states based on voice characteristics. The key steps involved in the process are:

- **Feature Extraction:** Prosodic features such as pitch and intensity are extracted from the audio recordings using the Parselmouth library.
- **Model Training:** The dataset is split into training and testing sets, and the SVM model is trained using the training set.
- **Emotion Prediction:** When a user records audio, the model predicts the emotional state based on the extracted features.

## Getting Started
### Prerequisites
- Python 3.x
- Flask
- Required Python libraries (listed in `requirements.txt`)

**Navigate to the project directory:**
```bash 
cd emotiontracker
```

**Install required libraries:**
```bash
pip install -r requirements.txt
```

## Running the Emotion Detection System
1. Prepare your dataset:
    - Place your audio files organized by emotion in the specified directory.
    
2. Run the EmotionTracker application:
```bash
python emotion.py
```
3. Open your web browser and go to `http://127.0.0.1:5000/survey` to access the survey interface.

## Dataset
The dataset used in this project consists of audio files categorized by emotion. Ensure the directory structure is set up correctly for the model to train and make predictions.

## Acknowledgements
- Kaggle for providing datasets and inspiration.
- Various libraries and frameworks used in this project, including PyAudio, Flask, and scikit-learn.















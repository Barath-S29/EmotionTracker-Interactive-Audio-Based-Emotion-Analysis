"""
Temporary script to analyze audio files in Dataset/archive and organize them
into emotion-based folders based on ML predictions.

Usage: python organize_archive.py
"""

import os
import shutil
import numpy as np
import parselmouth
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Try to import tqdm for progress bar, fall back to simple iteration if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        return iterable

# Import functions from Emotion.py
def extract_prosody_features(audio_file):
    """Extract prosody features from an audio file."""
    try:
        # Load the audio file using parselmouth
        snd = parselmouth.Sound(audio_file)

        # Extract prosody features using the Praat pitch and intensity objects
        pitch = snd.to_pitch()
        intensity = snd.to_intensity()

        # Calculate mean and standard deviation of pitch
        pitch_values = pitch.selected_array['frequency']
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)

        # Calculate mean and standard deviation of intensity
        intensity_values = intensity.values[0]
        intensity_mean = np.mean(intensity_values)
        intensity_std = np.std(intensity_values)

        # Return the extracted features as a numpy array
        return np.array([pitch_mean, pitch_std, intensity_mean, intensity_std])
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def load_dataset(data_dir):
    """Load dataset and extract prosody features."""
    X = []  # List to store the extracted features
    y = []  # List to store the corresponding emotion labels

    print(f"Loading dataset from {data_dir}...")
    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_dir) and emotion != "archive":  # Skip archive folder
            print(f"  Processing {emotion} folder...")
            for audio_file in os.listdir(emotion_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(emotion_dir, audio_file)
                    try:
                        features = extract_prosody_features(audio_path)
                        if features is not None:
                            X.append(features)
                            y.append(emotion)
                    except Exception as e:
                        print(f"    Skipping {audio_file}: {e}")

    return np.array(X), np.array(y)

def train_svm_model(X_train, y_train):
    """Train SVM model."""
    print("Training SVM model...")
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)
    print("Model trained successfully!")
    return svm_classifier

def get_all_audio_files(archive_dir):
    """Get all audio files from archive directory recursively."""
    audio_files = []
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def organize_archive_files(archive_dir, output_dir, model, copy_files=True):
    """
    Analyze and organize archive files based on emotion predictions.
    
    Args:
        archive_dir: Path to archive folder
        output_dir: Path to output directory (Dataset folder)
        model: Trained SVM model
        copy_files: If True, copy files; if False, move files
    """
    # Get all audio files from archive
    audio_files = get_all_audio_files(archive_dir)
    print(f"\nFound {len(audio_files)} audio files in archive.")
    
    # Statistics
    emotion_counts = {}
    failed_files = []
    
    print("\nAnalyzing and organizing files...")
    for audio_file in tqdm(audio_files, desc="Processing"):
        try:
            # Extract features
            features = extract_prosody_features(audio_file)
            
            if features is None:
                failed_files.append(audio_file)
                continue
            
            # Predict emotion
            predicted_emotion = model.predict([features])[0]
            
            # Create emotion folder if it doesn't exist
            emotion_dir = os.path.join(output_dir, predicted_emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Get filename
            filename = os.path.basename(audio_file)
            
            # Handle duplicate filenames by adding parent folder name
            dest_path = os.path.join(emotion_dir, filename)
            if os.path.exists(dest_path):
                # Add parent folder name to avoid conflicts
                parent_folder = os.path.basename(os.path.dirname(audio_file))
                name, ext = os.path.splitext(filename)
                filename = f"{parent_folder}_{name}{ext}"
                dest_path = os.path.join(emotion_dir, filename)
            
            # Copy or move file
            if copy_files:
                shutil.copy2(audio_file, dest_path)
            else:
                shutil.move(audio_file, dest_path)
            
            # Update statistics
            emotion_counts[predicted_emotion] = emotion_counts.get(predicted_emotion, 0) + 1
            
        except Exception as e:
            print(f"\nError processing {audio_file}: {e}")
            failed_files.append(audio_file)
    
    # Print summary
    print("\n" + "="*50)
    print("ORGANIZATION SUMMARY")
    print("="*50)
    print(f"\nTotal files processed: {len(audio_files)}")
    print(f"Successfully organized: {len(audio_files) - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    print("\nEmotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} files")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for failed_file in failed_files[:10]:  # Show first 10
            print(f"  - {failed_file}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print("\n" + "="*50)
    if copy_files:
        print("Files have been COPIED to emotion folders.")
        print("Original files in archive remain unchanged.")
    else:
        print("Files have been MOVED to emotion folders.")
        print("Original files in archive have been removed.")
    print("="*50)

def main():
    """Main function."""
    # Configuration
    dataset_dir = "Dataset"
    archive_dir = os.path.join(dataset_dir, "archive")
    output_dir = dataset_dir
    
    # Check if archive exists
    if not os.path.exists(archive_dir):
        print(f"Error: Archive directory not found: {archive_dir}")
        return
    
    # Step 1: Load and train model on existing dataset
    print("="*50)
    print("STEP 1: Training Model")
    print("="*50)
    X, y = load_dataset(dataset_dir)
    
    if len(X) == 0:
        print("\nERROR: No training data found!")
        print("Please ensure you have audio files in emotion folders:")
        print("  - Dataset/Angry/")
        print("  - Dataset/Fear/")
        print("  - Dataset/Happiness/")
        print("  - Dataset/Neutral/")
        print("  - Dataset/Sad/")
        return
    
    print(f"\nLoaded {len(X)} training samples.")
    
    # Split dataset
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_svm_model(X_train, y_train)
    
    # Step 2: Organize archive files
    print("\n" + "="*50)
    print("STEP 2: Organizing Archive Files")
    print("="*50)
    
    # Ask user preference
    print("\nChoose an option:")
    print("1. COPY files to emotion folders (keeps originals in archive)")
    print("2. MOVE files to emotion folders (removes from archive)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    copy_files = choice != "2"
    
    # Confirm
    action = "copy" if copy_files else "move"
    print(f"\nYou chose to {action} files.")
    confirm = input("Proceed? (yes/no, default=yes): ").strip().lower()
    
    if confirm and confirm not in ['yes', 'y', '']:
        print("Operation cancelled.")
        return
    
    # Organize files
    organize_archive_files(archive_dir, output_dir, model, copy_files=copy_files)
    
    print("\nScript completed!")

if __name__ == "__main__":
    main()

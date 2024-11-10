import tkinter as tk
from tkinter import filedialog, messagebox
import pyaudio
import wave
import threading
from keras.models import load_model
import numpy as np
import librosa
import time

# Load the saved model once at the start
model = load_model("emotion_detection_model.h5")

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection through Voice")
        self.root.geometry("400x300")
        
        # Instruction Label
        instruction_label = tk.Label(root, text="Upload or Record a voice note to detect emotion", font=("Arial", 12))
        instruction_label.pack(pady=10)
        
        # Upload Section
        upload_frame = tk.Frame(root)
        upload_frame.pack(pady=10)
        
        upload_button = tk.Button(upload_frame, text="Upload Voice Note", command=self.upload_file)
        upload_button.pack()
        
        # Record Section
        record_frame = tk.Frame(root)
        record_frame.pack(pady=10)
        
        self.record_button = tk.Button(record_frame, text="Record", command=self.start_recording)
        self.record_button.pack(side="left", padx=5)
        
        self.stop_button = tk.Button(record_frame, text="Stop", command=self.stop_recording, state="disabled")
        self.stop_button.pack(side="right", padx=5)
        
        # Result Display
        self.result_label = tk.Label(root, text="Detected Emotion: ", font=("Arial", 12))
        self.result_label.pack(pady=20)
        
        # Recording Attributes
        self.recording = False
        self.audio_frames = []
        
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            # Process the uploaded file to detect emotion
            self.process_audio(file_path)
    
    def start_recording(self):
        self.recording = True
        self.audio_frames = []
        
        # Disable record button, enable stop button
        self.record_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start recording in a new thread to avoid blocking GUI
        threading.Thread(target=self.record_audio).start()
    
    def stop_recording(self):
        self.recording = False
        self.record_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        # Save the recording as a file
        file_path = "recorded_voice.wav"
        self.save_audio(file_path)
        
        # Process the recorded file to detect emotion
        self.process_audio(file_path)
    
    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        
        while self.recording:
            data = stream.read(1024)
            self.audio_frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def save_audio(self, file_path):
        p = pyaudio.PyAudio()
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()

    def extract_mfcc(self, filename, n_mfcc=40):
        # Function to extract MFCC features as in your training pipeline
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        return mfcc

    def detect_emotion(self, file_path):
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(file_path)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Reshape for model input
        mfcc_features = np.expand_dims(mfcc_features, axis=-1)    
    
        prediction = model.predict(mfcc_features)
        emotion_index = np.argmax(prediction)
    
        # Map the output to emotion labels (adjust as per your model's training labels)
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
        detected_emotion = emotion_labels[emotion_index]
    
        return detected_emotion
    
    def process_audio(self, file_path):
        # Here, integrate the emotion detection model
        detected_emotion = self.detect_emotion(file_path)
        
        # Check for non-female voice
        if not self.is_female_voice(file_path):
            messagebox.showerror("Error", "Please upload a female voice.")
            return
        
        # Display the detected emotion
        self.result_label.config(text=f"Detected Emotion: {detected_emotion}")
    
    def is_female_voice(self, file_path):
        # Implement logic to check if the voice is female.
        # Placeholder return True. Replace with actual gender detection logic.
        return True

# Create and run the application
root = tk.Tk()
app = EmotionDetectionApp(root)
root.mainloop()

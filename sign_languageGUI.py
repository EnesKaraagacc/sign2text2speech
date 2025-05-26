"""
Sign Language to Text and Speech Translator

This application provides a PyQt5-based interface to predict Turkish sign language gestures
from videos using a trained 1D CNN model, and convert them into speech. It includes:
- Real-time video display
- Feature extraction using MediaPipe Holistic
- Prediction using a pre-trained CNN model
- Text display and voice synthesis via gTTS

"""

import sys
import os
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QFrame, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer
from gtts import gTTS
from playsound import playsound
from scipy.interpolate import interp1d
import mediapipe as mp
import uuid

# -------------------- CNN Model Definition --------------------
class SignLanguageCNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(SignLanguageCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 45)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -------------------- Wrapper for Inference --------------------
class DummySignModel:
    def __init__(self):
        self.input_dim = 150
        self.seq_len = 30
        self.num_classes = 45

        self.model = SignLanguageCNN1D(self.input_dim, self.num_classes)
        self.model.load_state_dict(torch.load("best_sign_language_model_cnn.pth", map_location="cpu"))
        self.model.eval()

        self.encoder = joblib.load("label_encoder_cleaned.pkl")

    def predict(self, input_seq):
        x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            idx = torch.argmax(logits, dim=1).item()
            return str(self.encoder.inverse_transform([idx])[0])

# -------------------- Feature Extraction using MediaPipe --------------------
mp_holistic = mp.solutions.holistic

def extract_features_from_video(video_path, max_frames=30):
    holistic = mp_holistic.Holistic(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    landmarks_seq = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        frame_landmarks = []

        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0] * 33 * 2)

        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0] * 21 * 2)

        # Right hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y])
        else:
            frame_landmarks.extend([0] * 21 * 2)

        landmarks_seq.append(frame_landmarks)

    cap.release()
    holistic.close()

    if len(landmarks_seq) == 0:
        return np.zeros((max_frames, 150))

    x_old = np.linspace(0, 1, num=len(landmarks_seq))
    x_new = np.linspace(0, 1, num=max_frames)
    interpolator = interp1d(x_old, np.array(landmarks_seq), axis=0, kind='linear', fill_value='extrapolate')
    return interpolator(x_new)

# -------------------- GUI Application --------------------
class TTSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤟 Turkish Sign Language → Text → Speech")
        self.setGeometry(300, 100, 1200, 850)
        self.setStyleSheet("background-color: #1e1e2f; color: white;")

        self.model = DummySignModel()
        self.loaded_input = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        layout = QVBoxLayout()

        title = QLabel("SIGN LANGUAGE INTERPRETER")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.video_label = QLabel("Video preview will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1000, 600)
        self.video_label.setStyleSheet("border: 2px solid #888; background-color: #222;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        video_control_layout = QHBoxLayout()
        self.start_button = QPushButton("▶️ Start")
        self.start_button.clicked.connect(self.start_video)
        video_control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏸ Stop")
        self.stop_button.clicked.connect(self.stop_video)
        video_control_layout.addWidget(self.stop_button)
        layout.addLayout(video_control_layout)

        layout.addSpacing(30)

        self.label = QLabel("📝 Predicted Text:")
        self.label.setFont(QFont("Segoe UI", 22))
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.text_edit = QTextEdit()
        self.text_edit.setFont(QFont("Segoe UI", 48))
        self.text_edit.setStyleSheet("background-color: #2b2b3d; color: white;")
        self.text_edit.setAlignment(Qt.AlignCenter)
        self.text_edit.setMinimumHeight(220)
        layout.addWidget(self.text_edit)

        button_layout = QHBoxLayout()
        self.load_button = QPushButton("📂 Load .npy")
        self.load_button.clicked.connect(self.dosya_yukle)
        button_layout.addWidget(self.load_button)

        self.predict_button = QPushButton("📊 Predict")
        self.predict_button.clicked.connect(self.tahmin_et)
        button_layout.addWidget(self.predict_button)

        self.video_button = QPushButton("🎥 Load Video")
        self.video_button.clicked.connect(self.video_yukle_ve_tahmin)
        button_layout.addWidget(self.video_button)

        self.ses_button = QPushButton("🔊 Speak")
        self.ses_button.clicked.connect(self.seslendir)
        button_layout.addWidget(self.ses_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                                  rgb_frame.shape[1] * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()

    def start_video(self):
        if self.cap and self.cap.isOpened():
            self.timer.start(30)

    def stop_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.video_label.clear()
        self.video_label.setText("Video preview will appear here")

    def dosya_yukle(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .npy file", "", "NumPy files (*.npy)")
        if not file_path:
            return
        try:
            data = np.load(file_path)
            if data.shape != (30, 150):
                QMessageBox.warning(self, "Invalid Input", f"Expected shape (30,150) but got {data.shape}")
                return
            self.loaded_input = data
            QMessageBox.information(self, "Success", "Feature data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error:\n{e}")

    def tahmin_et(self):
        if self.loaded_input is None:
            QMessageBox.warning(self, "No Input", "Please load a .npy file first.")
            return
        try:
            prediction = self.model.predict(self.loaded_input)
            self.text_edit.setPlainText(prediction)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Error:\n{e}")

    def video_yukle_ve_tahmin(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        try:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open the selected video file.")
                return
            self.timer.start(30)
            features = extract_features_from_video(file_path, max_frames=30)
            prediction = self.model.predict(features)
            self.text_edit.setPlainText(prediction)
            self.seslendir()
        except Exception as e:
            QMessageBox.critical(self, "Video Error", f"Error processing video:\n{e}")

    def seslendir(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Text", "Please enter or predict some text first.")
            return
        try:
            filename = f"tmp_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang='tr')
            tts.save(filename)
            playsound(filename)
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Could not generate speech:\n{e}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TTSApp()
    window.show()
    sys.exit(app.exec_())
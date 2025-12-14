import pandas as pd
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import numpy as np
import cv2
import imutils
import datetime
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from cv2 import imwrite
import HandDataCollecter
import mediapipe as mp

local_path = os.path.dirname(os.path.realpath('__file__'))
data_path = os.path.join(local_path, 'data.csv')
print(data_path)

df = pd.read_csv(data_path)
print(df)

units_in_data = 28
titles = ["unit-" + str(i) for i in range(units_in_data)]
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
clf = RandomForestClassifier(n_estimators=30)
clf.fit(X_train, y_train)

def create_model_1(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_2(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_3(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

data = pd.read_csv('data.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

labels = list(set(y))
label_map = {label: idx for idx, label in enumerate(labels)}
y = np.array([label_map[label] for label in y])
y_cat = to_categorical(y)

input_shape = (X_train.shape[1],)
num_classes = y_cat.shape[1]

model_1 = create_model_1(input_shape, num_classes)
model_2 = create_model_2(input_shape, num_classes)
model_3 = create_model_3(input_shape, num_classes)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def get_prediction(image):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        ImageData = HandDataCollecter.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        image = ImageData['image']
        prediction = clf.predict([DistanceData])
        return prediction[0]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    SpelledWord = ""

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        try:
            SpelledWord = get_prediction(image)
            print(SpelledWord)

            if SpelledWord == 'Women Safety Call':
                fromaddr = "sivarishi.cr7@gmail.com"
                toaddr = "sivarishi.cr7@gmail.com"
                imwrite("img.png", image)

                address = "SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India"
                maps_link = "https://www.google.com/maps?q=12.82506,80.04507"

                print("🚨 Emergency Detected")
                print("📍 Address:", address)
                print("📌 Maps Link:", maps_link)

                msg = MIMEMultipart()
                msg['From'] = fromaddr
                msg['To'] = toaddr
                msg['Subject'] = "🚨 WOMEN SAFETY CALL"

                body = f"""WOMEN SAFETY CALL TRIGGERED!

📍 Address: {address}
🗺️ Google Maps: {maps_link}
🕒 Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                msg.attach(MIMEText(body, 'plain'))

                filename = "img.jpg"
                attachment = open("img.png", "rb")
                p = MIMEBase('application', 'octet-stream')
                p.set_payload(attachment.read())
                encoders.encode_base64(p)
                p.add_header('Content-Disposition', f"attachment; filename= {filename}")
                msg.attach(p)

                # Send email
                s = smtplib.SMTP('smtp.gmail.com', 587)
                s.starttls()
                s.login(fromaddr, "yvho lsxu wjny ubct")  # Use your app-specific password
                s.sendmail(fromaddr, toaddr, msg.as_string())
                s.quit()

                print("📧 Emergency Email Sent!")
            else:
                print("❌ EMERGENCY not detected")

            cv2.putText(image, SpelledWord, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (124, 252, 0), 5, cv2.LINE_AA)
        except Exception as e:
            print("Error during prediction:", str(e))
            pass

        cv2.imshow('frame', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to break
            break

    cap.release()
    cv2.destroyAllWindows()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import cv2
import HandDataCollecter
import mediapipe as mp
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
########Initialise random forest

local_path = (os.path.dirname(os.path.realpath('__file__')))

file_name = ('data1.csv')  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)
df = pd.read_csv(r'' + data_path)

print(df)

units_in_data = 28  # no. of units in data

titles = []
for i in range(units_in_data):
    titles.append("unit-" + str(i))
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=30)  # random forest
clf.fit(X_train, y_train)
def create_model_1(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))  # Add dropout layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_2(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))  # Add dropout layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_3(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))  # Add dropout layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# Load data
data = pd.read_csv('data.csv')

# Extract features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Convert labels to numerical format
labels = list(set(y))  # Unique labels
label_map = {label: idx for idx, label in enumerate(labels)}
y = np.array([label_map[label] for label in y])

# Convert labels to one-hot encoding
y_cat = to_categorical(y)
# Create and train models
input_shape = (X_train.shape[1],)
num_classes = y_cat.shape[1]

model_1 = create_model_1(input_shape, num_classes)
model_2 = create_model_2(input_shape, num_classes)
model_3 = create_model_3(input_shape, num_classes)
#########Begin predictions
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
            # If loading a video, use 'break' instead of 'continue'.
            continue

        '''ImageData = HandDataCollecter.ImageToDistanceData(image, hands)
        DistanceData = ImageData['Distance-Data']
        image = ImageData['image']

        if cv2.waitKey(1) & 0xFF == 32:
            prediction = clf.predict([DistanceData])
            SpelledWord = str(prediction[0])
            #print(SpelledWord)'''

        try:
            SpelledWord = get_prediction(image)
            print(SpelledWord)
            cv2.putText(image, SpelledWord, (50, 50), 1, 2, 255)
            cv2.putText(image, SpelledWord, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
        except:
            pass

        cv2.imshow('frame', image)

        if cv2.waitKey(5) & 0xFF == 27:  # press escape to break
            break

    cap.release()
    cv2.destroyAllWindows()

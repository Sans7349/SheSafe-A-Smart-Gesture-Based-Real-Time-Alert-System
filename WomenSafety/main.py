import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('data1.csv')

# Extract features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Convert labels to numerical format
labels = list(set(y))  # Unique labels
label_map = {label: idx for idx, label in enumerate(labels)}
y = np.array([label_map[label] for label in y])

# Convert labels to one-hot encoding
y_cat = to_categorical(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define individual models
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

# Create and train models
input_shape = (X_train.shape[1],)
num_classes = y_cat.shape[1]

model_1 = create_model_1(input_shape, num_classes)
model_2 = create_model_2(input_shape, num_classes)
model_3 = create_model_3(input_shape, num_classes)

history_1 = model_1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)
history_2 = model_2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)
history_3 = model_3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)

# Predict with each model
preds_1 = model_1.predict(X_test)
preds_2 = model_2.predict(X_test)
preds_3 = model_3.predict(X_test)

# Evaluate individual model accuracies for weighting
def evaluate_model_accuracy(model, X_test, y_test_cat):
    preds = model.predict(X_test)
    preds_classes = np.argmax(preds, axis=1)
    y_test_classes = np.argmax(y_test_cat, axis=1)
    accuracy = accuracy_score(y_test_classes, preds_classes)
    return accuracy

accuracy_1 = evaluate_model_accuracy(model_1, X_test, y_test)
accuracy_2 = evaluate_model_accuracy(model_2, X_test, y_test)
accuracy_3 = evaluate_model_accuracy(model_3, X_test, y_test)

# Display individual model accuracies
print(f'Model 1 Accuracy: {accuracy_1:.4f}')
print(f'Model 2 Accuracy: {accuracy_2:.4f}')
print(f'Model 3 Accuracy: {accuracy_3:.4f}')

# Calculate weighted average for ensemble predictions based on individual model accuracies
total_accuracy = accuracy_1 + accuracy_2 + accuracy_3
weights = [accuracy_1 / total_accuracy, accuracy_2 / total_accuracy, accuracy_3 / total_accuracy]

ensemble_preds = (weights[0] * preds_1 + weights[1] * preds_2 + weights[2] * preds_3)
ensemble_preds_classes = np.argmax(ensemble_preds, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate ensemble accuracy
ensemble_accuracy = accuracy_score(y_test_classes, ensemble_preds_classes)
print(f'Weighted Ensemble Accuracy: {ensemble_accuracy:.4f}')

import matplotlib.pyplot as plt

# Plot training and validation accuracy for each model
def plot_model_history(history, model_name):
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{model_name} Training and Validation Accuracy')

# Plot history for Model 1
plt.figure(figsize=(10, 6))
plot_model_history(history_1, "Model 1")
plt.show()

# Plot history for Model 2
plt.figure(figsize=(10, 6))
plot_model_history(history_2, "Model 2")
plt.show()

# Plot history for Model 3
plt.figure(figsize=(10, 6))
plot_model_history(history_3, "Model 3")
plt.show()
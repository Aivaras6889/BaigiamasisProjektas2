import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import json
from preprocessing import train_edited_images, test_edited_images, labels_train, labels_test
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()

X = np.array(train_edited_images)
X = X / 255.0
y = np.array([int(label) for label in labels_train])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(96, 96, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten(name='flatten'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(43, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history_cnn = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[early_stop],
    verbose=2
)

model.save("cnn_model.keras")
plot_training_history(history_cnn)

X_test = np.array(test_edited_images)
X_test = X_test / 255.0
test_predictions_prob = model.predict(X_test, verbose=0)
test_predictions_class = np.argmax(test_predictions_prob, axis=1)

print(f"Final Training Loss: {history_cnn.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history_cnn.history['accuracy'][-1]:.4f}")
print(f"Final Validation Loss: {history_cnn.history['val_loss'][-1]:.4f}")
print(f"Final Validation Accuracy: {history_cnn.history['val_accuracy'][-1]:.4f}")

val_predictions_prob = model.predict(X_val, verbose=0)
val_predictions_class = np.argmax(val_predictions_prob, axis=1)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, val_predictions_class, average='weighted')

print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")

np.save("test_predictions.npy", test_predictions_class)
import zipfile
import os
with zipfile.ZipFile('RomeWeather.zip', 'r') as zip_ref:
        zip_ref.extractall('/content')
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Define data path
train_path = '/content/Rome Weather/Train'
test_path = '/content/Rome Weather/Test'

# Categories : name of the folders
categories = ['Cloudy', 'Foggy', 'Rainy', 'Snowy', 'Sunny']

def load_data(data_path, categories, img_size=(128, 128)):
    data = []
    labels = []

    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Load the picture with OpenCV
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error with the picture : {img_name}: {e}")

    return np.array(data), np.array(labels)

# Charging training data
x_train, y_train = load_data(train_path, categories)

# Charging test data
x_test, y_test = load_data(test_path, categories)

# Normalize pixels between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Transforme labels in one-hot encoding
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_image(image_path, model, categories):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]

prediction = predict_image('/content/Rome Weather/Test/Cloudy/download (1).jpg', model, categories)
print(f"Predicted category is : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Cloudy/download (2).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Foggy/download (1).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Foggy/download (7).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Rainy/download (4).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Rainy/download (5).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Snowy/download (6).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Snowy/download (11).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Sunny/images (8).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Sunny/images (6).jpg', model, categories)
print(f"Predicted category is  : {prediction}")

# Model with Dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Dropout après la première couche

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Dropout après la seconde couche

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout avant la dernière couche
    Dense(len(categories), activation='softmax')
])

# Compile the modele
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Re-training the model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Early stopping if the performances doesn't improve anymore
early_stopping = EarlyStopping(
    monitor='val_loss',  # Look for loss on validation set
    patience=3,  # Stop after 3 epochs without improvement
    restore_best_weights=True  # Restore the optimals weights
)

# Re-training with EarlyStopping
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,  # High number to observe the effects of EarlyStopping
    batch_size=32,
    callbacks=[early_stopping]
)

# Generate enhanced data
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Re-training with enhanced data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=[early_stopping]
)

# Get predictions
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Comute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# display with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.show()

# Generate a WordCloud6
predictions = np.argmax(model.predict(x_test), axis=1)
word_counts = {categories[i]: (predictions == i).sum() for i in range(len(categories))}
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Prediction's WordCloud")
plt.show()

# Get test's predictions
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Display classification report
print(classification_report(y_true, y_pred, target_names=categories))

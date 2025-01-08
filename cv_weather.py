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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Définir les chemins vers les données
train_path = '/content/Rome Weather/Train'
test_path = '/content/Rome Weather/Test'

# Les catégories : les noms des sous-dossiers
categories = ['Cloudy', 'Foggy', 'Rainy', 'Snowy', 'Sunny']

def load_data(data_path, categories, img_size=(128, 128)):
    data = []
    labels = []

    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_path, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Charger l'image avec OpenCV
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)  # Redimensionner
                data.append(img)
                labels.append(idx)  # Label numérique (0 pour earth, 1 pour saturn)
            except Exception as e:
                print(f"Erreur avec l'image {img_name}: {e}")

    return np.array(data), np.array(labels)

# Charger les données d'entraînement
x_train, y_train = load_data(train_path, categories)

# Charger les données de test
x_test, y_test = load_data(test_path, categories)

# Normaliser les pixels entre 0 et 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Transformer les labels en one-hot encoding
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')  # Len(categories) = 2 pour earth et saturn
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,  # Vous pouvez ajuster le nombre d'époques
    batch_size=32
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_image(image_path, model, categories):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour le batch
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]

prediction = predict_image('/content/Rome Weather/Test/Cloudy/download (1).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Cloudy/download (2).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Foggy/download (1).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Foggy/download (7).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Rainy/download (4).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Rainy/download (5).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Snowy/download (6).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Snowy/download (11).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Sunny/images (8).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

prediction = predict_image('/content/Rome Weather/Test/Sunny/images (6).jpg', model, categories)
print(f"La classe prédite est : {prediction}")

# Modèle avec Dropout
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

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Réentraînement du modèle
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Early stopping pour arrêter si la performance ne s'améliore plus
early_stopping = EarlyStopping(
    monitor='val_loss',  # Surveille la perte sur le set de validation
    patience=3,  # Stop après 3 époques sans amélioration
    restore_best_weights=True  # Restaure les poids optimaux
)

# Réentraînement avec EarlyStopping
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,  # Nombre élevé pour observer l'effet d'EarlyStopping
    batch_size=32,
    callbacks=[early_stopping]
)

# Générateur de données avec augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Réentraînement avec des données augmentées
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=10,
    callbacks=[early_stopping]
)

# Obtenir les prédictions
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculer la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Afficher avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()

# Générer un nuage de mots
predictions = np.argmax(model.predict(x_test), axis=1)
word_counts = {categories[i]: (predictions == i).sum() for i in range(len(categories))}
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)

# Afficher le nuage
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Nuage de Mots des Prédictions")
plt.show()

# Obtenir les prédictions sur le test
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Afficher le rapport de classification
print(classification_report(y_true, y_pred, target_names=categories))

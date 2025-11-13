import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# TPU Configuration
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

def get_device():
    return xm.xla_device()

device = get_device()

# Paths to dataset
DATASET_PATH = "/kaggle/input/plantvillage-dataset"
COLOR_PATH = os.path.join(DATASET_PATH, "color")

# Image size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation & Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    COLOR_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(
    COLOR_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Base CNN Feature Extractors
base_models = {
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
}

# Feature extraction and classification
def build_model(base_model):
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train each feature extraction model
models = {}
with strategy.scope():
    for name, base_model in base_models.items():
        print(f"Training {name} on TPU...")
        model = build_model(base_model)
        model.fit(train_generator, validation_data=val_generator, epochs=5)
        model.save(f"{name}_plant_disease_model_tpu.h5")
        models[name] = model

# Vision Transformer (ViT) Model
class PlantDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.classes = os.listdir(path)
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(path, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = PlantDataset(COLOR_PATH, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224")
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-large-patch16-224",
    num_labels=len(dataset.classes),
    ignore_mismatched_sizes=True
)
vit_model.to(device)

optimizer = torch.optim.AdamW(vit_model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(1):
    vit_model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = vit_model(images).logits
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

torch.save(vit_model.state_dict(), "vit_large_plant_disease_tpu.pth")

# Prediction function
def predict_plant_disease(image_path, model_name="ViT"):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    if model_name == "ViT":
        vit_model.eval()
        with torch.no_grad():
            outputs = vit_model(image).logits
            predicted_class = torch.argmax(outputs, dim=1).item()
            return dataset.classes[predicted_class]
    else:
        model = load_model(f"{model_name}_plant_disease_model_tpu.h5")
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        return train_generator.class_indices[np.argmax(predictions)]

# Example usage
image_path = "/kaggle/input/tom-early-blight/WhatsApp Image 2025-01-18 at 13.50.01_2af1dfb0.jpg"
print(predict_plant_disease(image_path, model_name="ViT"))

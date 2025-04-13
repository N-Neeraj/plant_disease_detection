import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB0, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# -------------------------------
# 🔥 GPU SETUP FOR MAX UTILIZATION
# -------------------------------
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"🔥 Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

# Optimize PyTorch GPU usage
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# TensorFlow Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(num_gpus)])
print(f"🌍 TensorFlow will use {strategy.num_replicas_in_sync} GPUs.")

# -------------------------------
# 🔥 DATASET & DATALOADERS
# -------------------------------
DATASET_PATH = "/kaggle/input/plantvillage-dataset"
COLOR_PATH = os.path.join(DATASET_PATH, "color")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32 * max(1, num_gpus)  # Scale batch size based on GPU count

if not os.path.exists(COLOR_PATH):
    print(f"⚠️ Dataset not found at {COLOR_PATH}! Please check your dataset path.")
else:
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        COLOR_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
    val_generator = datagen.flow_from_directory(
        COLOR_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

    # -------------------------------
    # 🔥 CNN MODELS WITH MULTI-GPU SUPPORT
    # -------------------------------
    base_models = {
        "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
        "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    }

    def build_model(base_model):
        with strategy.scope():
            base_model.trainable = False
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            output = Dense(train_generator.num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    models = {}
    history_dict = {}

    for name, base_model in base_models.items():
        print(f"🚀 Training {name} on {num_gpus} GPUs...")
        model = build_model(base_model)
        history = model.fit(train_generator, validation_data=val_generator, epochs=1)
        model.save(f"{name}_plant_disease_model_multi_gpu.h5")
        models[name] = model
        history_dict[name] = history.history

    # 📊 Plotting Accuracy Graphs for CNNs
    for name, history in history_dict.items():
        plt.plot(history['accuracy'], label=f'{name} Accuracy')
    plt.title('CNN Models Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------
# 🔥 VISION TRANSFORMER (ViT) - MULTI-GPU
# -------------------------------
class PlantDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.classes = sorted(os.listdir(path))
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
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = PlantDataset(COLOR_PATH, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(dataset.classes),
    ignore_mismatched_sizes=True
)

if num_gpus > 1:
    vit_model = torch.nn.DataParallel(vit_model)

vit_model.to(device)

optimizer = torch.optim.AdamW(vit_model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

vit_accuracies = []

for epoch in range(1):
    vit_model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc=f"🔥 Epoch {epoch+1}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = vit_model(images).logits
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(loss=loss.item())
    accuracy = correct / total
    vit_accuracies.append(accuracy)
    print(f"🔥 Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, Accuracy: {accuracy}")

torch.save(vit_model.state_dict(), "vit_base_plant_disease_multi_gpu.pth")

# 📊 Plot ViT accuracy graph
plt.plot(range(1, len(vit_accuracies)+1), vit_accuracies, label="ViT Accuracy", marker='o')
plt.title("Vision Transformer Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 🔥 PREDICTION FUNCTION (with image display)
# -------------------------------
def predict_plant_disease(image_path, model_name="ViT"):
    if not os.path.exists(image_path):
        return "⚠️ Image file not found!"

    # Display the input image
    img_disp = Image.open(image_path).convert('RGB')
    plt.imshow(img_disp)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    if model_name == "ViT":
        image_tensor = transform(img_disp).unsqueeze(0).to(device)
        vit_model.eval()
        with torch.no_grad():
            outputs = vit_model(image_tensor).logits
            predicted_class = torch.argmax(outputs, dim=1).item()
            return dataset.classes[predicted_class]
    else:
        model = load_model(f"{model_name}_plant_disease_model_multi_gpu.h5")
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions)
        class_indices = {v: k for k, v in train_generator.class_indices.items()}
        return class_indices[predicted_class_idx]

# 🔮 Run prediction and show result
image_path = "/kaggle/input/grape-black-rot/WhatsApp Image 2025-04-04 at 20.14.24_78b031ab.jpg"
print("Prediction:", predict_plant_disease(image_path, model_name="ViT"))

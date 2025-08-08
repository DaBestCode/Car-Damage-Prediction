import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os



trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_model.pth"
GOOGLE_DRIVE_FILE_ID = "1rryu5it14lUf-MmUDjrxlCvjpwVDsMTK"  # file ID
MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

def download_model():
    """Download model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found. Downloading from {MODEL_URL}...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print(" Model downloaded successfully.")

# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
def load_model():
    """Load the trained model from file."""
    global trained_model
    if trained_model is None:
        download_model()
        model = CarClassifierResNet(num_classes=len(class_names))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        trained_model = model
    return trained_model


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]






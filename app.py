from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import yaml
from models.ResnetFFN import ResnetFFN
import io
import os
import pandas as pd  # Import the pandas library

from dotenv import load_dotenv

load_dotenv()

with open(os.getenv("CONFIG_DIR"), "r") as config_file:
    config_model = yaml.safe_load(config_file)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(type(device))

# Load a pretrained model_state_dict under ./pretrained directory
model_state_dict = torch.load("pretrained/model_state_dict_mse_0_0045.pth")

# Create an instance of the model
model = ResnetFFN(
    device,
    config_model["Model"]["num_outputs"],
    config_model["Model"]["batch_first"],
    config_model["Model"]["conv_channel"],
    config_model["Model"]["fc_hidden_dims"],
)

model.load_state_dict(model_state_dict)
model.to(device)
model.eval()

app = Flask(__name__)

IMG_H = 224
IMG_W = 224

# Set predetermined mean and std values
mean_images = np.array([122.09624237, 123.38567456, 120.75862292]) / 255.0
std_images = np.array([61.13438223, 62.09970917, 65.60647365]) / 255.0

# Resize images, convert to torch.Tensor and normalize the dataset regarding predetermined mean and std values
transform = transforms.Compose(
    [
        transforms.Resize((IMG_H, IMG_W)),  # Resize images to (IMG_H x IMG_W)
        transforms.ToTensor(),
        transforms.Normalize(mean_images, std_images),
    ]
)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if "file_name" not in data:
        return jsonify({"error": "No file_name part"}), 400

    file_name = data["file_name"]
    if file_name == "":
        return jsonify({"error": "No selected file_name"}), 400

    try:
        img_path = os.path.join("data/CodingChallenge_v2/imgs/", file_name + ".jpg")
        img = Image.open(img_path)
        img_tensor = (
            transform(img).unsqueeze(0).to(device)
        )  # Add batch dimension and move to device

        with torch.no_grad():
            print("\nMODEL OUTPUT:")
            output = model(img_tensor)
            print(output)

        # Find img_id in csv_file_path = "./data/CodingChallenge_v2/car_imgs_4000.csv" and print the perspective_score_hood and perspective_score_backdoor_left
        csv_file_path = "./data/CodingChallenge_v2/car_imgs_4000.csv"
        img_id = file_name  # Define the img_id variable
        df = pd.read_csv(csv_file_path)
        print("\nGROUND TRUTH:")
        gt_perspective_score_hood, gt_perspective_score_backdoor_left = df[
            df["filename"] == img_id + ".jpg"
        ][["perspective_score_hood", "perspective_score_backdoor_left"]].values[0]

        return jsonify(
            {
                "perspective_score_hood": output[0][0].item(),
                "perspective_score_hood_gt": gt_perspective_score_hood,
                "perspective_score_backdoor_left": output[0][1].item(),
                "perspective_score_backdoor_left_gt": gt_perspective_score_backdoor_left,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

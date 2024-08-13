## Getting Started

## Overview

The aim of this repository is to predict the perspective scores of different car parts and their availability in the given images. This project involves training a neural network model to analyze images of cars and provide perspective scores for specific parts such as the hood and left backdoor. The model can be used to assess the visibility and orientation of these parts in the images, which can be useful for various applications in automotive analysis and computer vision.

### Prerequisites

- Docker
- NVIDIA Docker (for GPU support)
- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/Perspective-Regressor.git
    cd Perspective-Regressor
    ```

2. **Install the required Python packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Running the Application

1. **Build the Docker image using the provided script:**

    ```sh
    ./build.sh
    ```

2. **Deploy the model using the provided script:**

    ```sh
    ./deploy_model4app.sh
    ```

### API Usage

The application exposes a REST API for making predictions. The API endpoint is `/predict`.

#### Request

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Body:**

    ```json
    {
        "file_name": "example_image"
    }
    ```

#### Response

- **Success Response:**

    ```json
    {
        "perspective_score_hood": 0.123,
        "perspective_score_trunk": 0.456
    }
    ```

- **Error Response:**

    ```json
    {
        "error": "Error message"
    }
    ```

### Training the Model from Scratch

You can also train the model from scratch using the provided Jupyter Notebook.

1. **Open `main.ipynb` in Jupyter Notebook or JupyterLab:**

    ```sh
    jupyter notebook main.ipynb
    ```

2. **Follow the instructions in the notebook to train the model.**

### Project Files

- **`app.py`:** The main Flask application file.
- **`build.sh`:** Shell script to build the Docker image.
- **`deploy_model4app.sh`:** Shell script to deploy the model using Docker.
- **`Dockerfile`:** Dockerfile to build the Docker image.
- **`main.ipynb`:** Jupyter Notebook for training the model from scratch.
- **`pretrained/`:** Pretrained model weights.
- **`data/`:** Directory containing the dataset.
- **`requirements.txt`:** List of required Python packages.

### Notes

- Ensure that the `CONFIG_DIR` environment variable is set to the path of your configuration file.
- The `data/imgs` directory should contain the images used for prediction.
- The `data/car_imgs_4000.csv` file should contain the ground truth perspective scores.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
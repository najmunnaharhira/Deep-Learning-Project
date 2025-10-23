🐶🐱 Dog vs Cat Classification using Transfer Learning (MobileNet V2)
]
![main image ](https://github.com/user-attachments/assets/bc9d4e90-7f7c-4ea4-84bf-77e95ce23430)

 ![Github Repo](https://img.shields.io/badge/GitHub-najmunnaharhira%2FDeep--Learning--Project-blue?logo=github)

🔹 Project Overview 🔹

This project demonstrates image classification using Transfer Learning, leveraging the MobileNet V2 pre-trained model to classify images of dogs and cats.
By using Transfer Learning, we reuse knowledge from a model trained on the large ImageNet dataset, achieving excellent accuracy on a smaller, domain-specific dataset.

🔹 Project Overview (Video) 🔹

🎥 Video Link : https://youtu.be/gBu9Y460SMg?si=BZfY81eBqrB3CDLd


🔹 Dataset 🔹

📂 Dogs vs Cats Dataset (Kaggle)

Contains 25,000 labeled images of dogs and cats used for training and validation.

🔹 Methodology 🔹

✅ Data Preprocessing:

Loaded and resized all images to 224×224 pixels

Normalized pixel values between 0 and 1

Labeled classes for binary classification (Dog/Cat)

✅ Model Architecture (Transfer Learning):

Used MobileNetV2 pre-trained on ImageNet

Froze base layers to retain learned features

Added custom dense layers for classification

✅ Training Details:

Optimizer: Adam

Loss Function: binary_crossentropy

Metrics: accuracy

Epochs: 5–10

✅ Evaluation:

Achieved up to 98% validation accuracy

Visualized accuracy & loss curves using Matplotlib

🔹 Tech Stack 🔹

💻 Development Environment:

Google Colab — for cloud-based notebook execution and GPU support

Python 3.x — programming language

🧠 Deep Learning Frameworks:

TensorFlow / Keras — model building, training, and evaluation

MobileNetV2 — pre-trained CNN model used for transfer learning

📊 Data Handling & Visualization:

NumPy — numerical operations

Pandas — data manipulation

Matplotlib & Seaborn — visualizing accuracy, loss, and predictions

🖼 Dataset:

Kaggle — Dogs vs Cats Dataset (25,000 labeled images of dogs and cats)

☁ Platform:

Google Drive Integration — for dataset storage and easy Colab access

🔹 Installation 🔹
git clone https://github.com/najmunnaharhira/Deep-Learning-Project.git
cd Deep-Learning-Project
pip install -r requirements.txt

🔹 How to Use 🔹

✅ Open the notebook DL_Project_Dog_vs_Cat_Classification_Transfer_Learning.ipynb in Google Colab
✅ Mount Google Drive for dataset access
✅ Run all cells sequentially
✅ The model will train and display accuracy & loss graphs
✅ Use the trained model to predict new dog or cat images

🔹 Results 🔹

📈 Validation Accuracy: ~98%
📉 Validation Loss: ~0.05

Image	Prediction
🐶 dog1.jpg	✅ Dog
🐱 cat1.jpg	✅ Cat
🔹 Future Improvement Ideas 🔹

✨ Implement real-time classification using OpenCV
✨ Add data augmentation for better generalization
✨ Experiment with other models (ResNet, EfficientNet)
✨ Deploy model using Streamlit or Flask Web App

🔹 References 🔹

Kaggle Dogs vs Cats Dataset

TensorFlow MobileNetV2

Keras Transfer Learning Guide

🔹 Screenshots 🔹


![image2](https://github.com/user-attachments/assets/e980c6b9-e6d6-41ca-914a-f22125bbd8b3)
![image 3](https://github.com/user-attachments/assets/6b82cef5-6995-4d88-8e45-94acaf4c3d54)
![image 1](https://github.com/user-attachments/assets/8c7cf141-5923-459d-9879-8bb544222769)




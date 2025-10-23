# 🐶🐱 Dog vs Cat Classification using Transfer Learning (MobileNet V2)

![main image](https://github.com/user-attachments/assets/6bc72e71-193d-4ef2-8987-f95d1fce0f09)


 ![Github Repo](https://img.shields.io/badge/GitHub-najmunnaharhira%2FDeep--Learning--Project-blue?logo=github)

## 🔹 Project Overview 🔹

This project demonstrates image classification using Transfer Learning, leveraging the MobileNet V2 pre-trained model to classify images of dogs and cats.
By using Transfer Learning, we reuse knowledge from a model trained on the large ImageNet dataset, achieving excellent accuracy on a smaller, domain-specific dataset.

## 🔹 Project Overview (Video) 🔹

[🎥 **Watch Project Video**](https://youtu.be/gBu9Y460SMg?si=BZfY81eBqrB3CDLd)


## 🔹 Dataset 🔹

📂 Dogs vs Cats Dataset (Kaggle)
 **https://www.kaggle.com/c/dogs-vs-cats/data**

Contains 25,000 labeled images of dogs and cats used for training and validation.

## 🔹 Methodology 🔹

✅ **Data Preprocessing**:

Loaded and resized all images to 224×224 pixels

Normalized pixel values between 0 and 1

Labeled classes for binary classification (Dog/Cat)

✅**Model Architecture (Transfer Learning)**:

Used MobileNetV2 pre-trained on ImageNet

Froze base layers to retain learned features

Added custom dense layers for classification

✅ **Training Details**:

Optimizer: Adam

Loss Function: binary_crossentropy

Metrics: accuracy

Epochs: 5–10

✅ **Evaluation**:

Achieved up to 98% validation accuracy

Visualized accuracy & loss curves using Matplotlib

## 🔹Tech Stack 🔹

💻 **Development Environment**:

Google Colab — for cloud-based notebook execution and GPU support

Python 3.x — programming language

🧠 **Deep Learning Frameworks**:

TensorFlow / Keras — model building, training, and evaluation

MobileNetV2 — pre-trained CNN model used for transfer learning

📊 **Data Handling & Visualization**:

NumPy — numerical operations

Pandas — data manipulation

Matplotlib & Seaborn — visualizing accuracy, loss, and predictions

🖼 **Dataset**:

Kaggle — Dogs vs Cats Dataset (25,000 labeled images of dogs and cats)

☁ **Platform**:

Google Drive Integration — for dataset storage and easy Colab access

## 🔹 Installation 🔹
git clone https://github.com/najmunnaharhira/Deep-Learning-Project.git
cd Deep-Learning-Project
pip install -r requirements.txt

## 🔹 How to Use 🔹

✅ Open the notebook DL_Project_Dog_vs_Cat_Classification_Transfer_Learning.ipynb in **Google Colab**

✅ Mount Google Drive for dataset access

✅ Run all cells sequentially

✅ The model will train and display accuracy & loss graphs

✅ Use the trained model to predict new dog or cat images


## 🔹 Results 🔹

**📈 Validation Accuracy: ~98%**
**📉 Validation Loss: ~0.05**

## Image	Prediction
**🐶 dog1.jpg	✅ Dog**

**🐱 cat1.jpg	✅ Cat**

## 🔹 Future Improvement Ideas 🔹

✨ Implement real-time classification using OpenCV
✨ Add data augmentation for better generalization
✨ Experiment with other models (ResNet, EfficientNet)
✨ Deploy model using Streamlit or Flask Web App

## 🔹 References 🔹

**Kaggle Dogs vs Cats Dataset**

**TensorFlow MobileNetV2**

**Keras Transfer Learning Guide**
## 🔹 Screenshots 🔹
 ![img 1](https://github.com/user-attachments/assets/899fedad-86de-4fc0-9f8f-12208ee11812)
 ![img 2](https://github.com/user-attachments/assets/93eee8e3-8125-47ff-a41b-0b98ff13f7e9)
 ![img 3](https://github.com/user-attachments/assets/3575223a-517a-443e-a65f-5ad17c777579)
 ![img 4](https://github.com/user-attachments/assets/b1f7f84f-a964-4fb3-8e96-a7476d55b4d0)











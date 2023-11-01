# Bias Analysis in Federated Learning for Heterogeneous Sensors

## Motivation
Federated learning aims to train a machine learning model collaboratively while keeping the data of the participants private. Different types of sensitive data like images, audio, text, and sensor data can be collected to train models, but this information can be exploited by adversaries which would impact the privacy of parties. As adversarial capabilities continue to increase and attacks on machine learning models improve, the threat of an adversary recovering confidential data used to train models is increasing. Federated learning serves as a way to utilize data from multiple sources for a single learning application allowing for multiple parties to keep their data private.

Our group has experience dealing with machine learning models, specifically image recognition and using CUDA-enabled GPUs. We also have experience in security engineering and 547. This will also allow us to gain further knowledge in applying machine-learning approaches to our SDP projects.
 
## Design Goals
- Variations in feature distribution in federating learning leads to biases. Our goal is to model this bias towards different groups.
  
## Deliverables
- Understanding different federated learning techniques including - Federated averaging, Tilted Empirical Risk Minimization, and agnostic federated learning.
- Use different datasets for training and assessing two federated learning techniques.
- Examine the enhancement in the variance of accuracy among individual client groups when employing various federated learning techniques.

## System Blocks
Provided Data -> Federated Learning (using Laptop with CUDA-enabled GPU)

## HW/SW Requirements
- Python
- Laptop with CUDA-enabled GPU

## Team Members Responsibilities
- Brayden Bergeron - Setup, Software - Communication-Efficient Learning of Deep Networks from Decentralized Data
- Khushali Shah - Networking, algorithm designs - Agnostic Federated Learning
- Jenny Utstein - Writing, Research - Tilted Empirical Risk Minimization

## Project Timeline
Our team will work on the project for at least 1-2 hours in a week
- Complete reviewing references
- Gain an understanding of federated learning
- Setup Python with CUDA enabled GPU
- Begin analysis of datasets and analyzing model accuracy
 

## References
- Communication-Efficient Learning of Deep Networks from Decentralized Data (https://arxiv.org/abs/1602.05629)
- Tilted Empirical Risk Minimization (https://openreview.net/pdf?id=K5YasWXZT3O)
- Agnostic Federated Learning (https://arxiv.org/pdf/1902.00146.pdf)
- FedAvg (https://github.com/alexbie98/fedavg)
- TERM (https://github.com/litian96/TERM)
- AFL (https://github.com/YuichiNAGAO/agnostic_federated_learning)
- CIFAR-10 (https://www.kaggle.com/c/cifar-10/data)
- FashionMNIST (https://github.com/zalandoresearch/fashion-mnist)

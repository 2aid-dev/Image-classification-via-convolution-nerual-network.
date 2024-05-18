# Image-classification-via-convolution-nerual-network.

After downloading and installing the trial environment requirements.
<br />
***The model will***:
- A dataset of images is loaded and transformations are applied to it.
- A ResNet-18 model is built and loaded.
- The loss function and optimizer are defined.
- The model is trained and tested over multiple epochs, and the best-performing model is saved based on accuracy.

<img src="https://github.com/2aid-dev/Image-classification-via-convolution-nerual-network./assets/42585484/2c21cbdf-c608-4fa2-b09d-275ec5d4caaa" alt="drawing" width="100%" height="600"/>
--- 

# So let's do the homework.

## Here are ways to optimize the following experience: 👇🏼
https://github.com/WhynotChen0105/Experiment-1/tree/main

-----

<details>
  <summary> <b> Description of the experiment </b> </summary>
  
  In this experiment, we aim to classify images using a Convolutional Neural Network (CNN). The steps involved in this experiment are as follows:

  #### 1. Data Preparation
  - **Dataset Collection**: Gather a labeled dataset of images. Common datasets include CIFAR-10, MNIST, or a custom dataset.
  - **Data Preprocessing**: Normalize the images, resize them to a consistent size, and split the dataset into training, validation, and test sets.

  #### 2. Model Design
  - **Architecture Selection**: Choose an appropriate CNN architecture. For example, you can start with a simple model or use pre-trained models like ResNet50, VGG16, or MobileNet.
  - **Layer Configuration**: Configure the layers of the CNN, including convolutional layers, pooling layers, dropout layers, and fully connected layers.

  #### 3. Training the Model
  - **Loss Function**: Select a suitable loss function such as categorical cross-entropy for multi-class classification.
  - **Optimizer**: Choose an optimizer like Adam, SGD, or RMSprop to minimize the loss function.
  - **Data Augmentation**: Apply data augmentation techniques such as rotation, flipping, and zooming to increase the variability of the training data.
  - **Training Process**: Train the CNN on the training dataset, monitor the performance on the validation set, and adjust hyperparameters as needed.

  #### 4. Model Evaluation
  - **Performance Metrics**: Evaluate the model using metrics such as accuracy, precision, recall, and F1-score on the test dataset.
  - **Confusion Matrix**: Analyze the confusion matrix to understand the classification performance for each class.

  #### 5. Fine-Tuning and Optimization
  - **Hyperparameter Tuning**: Experiment with different hyperparameters like learning rate, batch size, and the number of epochs.
  - **Transfer Learning**: Utilize pre-trained models and fine-tune them on the specific dataset to improve performance.
  - **Regularization Techniques**: Implement techniques like dropout and batch normalization to prevent overfitting.

  #### 6. Deployment
  - **Model Export**: Save the trained model in a suitable format for deployment.
  - **Inference Pipeline**: Develop an inference pipeline to classify new images in real-time or batch mode.
  - **Deployment Environment**: Deploy the model to a production environment such as a cloud service, edge device, or mobile application.

  By following these steps, the experiment aims to achieve high accuracy in image classification tasks using CNNs.
</details>


***


# Improving the performance of your model can be achieved through a variety of strategies. Here are some ways to enhance your model:

### 1. **Improving Training Data**

#### Data Augmentation
Using data augmentation techniques can help the model generalize better.

```python
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    torchvision.transforms.ToTensor()
])
```

#### Improving Test Data Transformations
Apply appropriate transformations to the test data, but avoid excessive augmentation here.

```python
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor()
])
```

### 2. **Improving the Model Architecture**

#### Using a More Complex Model
Using a more complex or modified model can help improve performance. You can try models like ResNet50 or even use transfer learning techniques.

```python
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
```

### 3. **Improving the Loss Function and Optimizer**

#### Enhancing the Loss Function
If the current loss function is insufficient, you can try other loss functions or a combination of them.

#### Enhancing the Optimizer
Experiment with other optimizers such as Adam, or use optimization techniques like learning rate scheduling.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

### 4. **Using Training Optimization Techniques**

#### Using Pretraining
Using pretrained models and fine-tuning them can significantly improve model performance.

```python
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# Fine-tune the last layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 102)  # Assuming 102 classes in the Flowers102 dataset
```

#### Multi-GPU Training
Use multi-GPU training techniques to distribute the workload across several GPUs.

```python
model = torch.nn.DataParallel(model)
```

### 5. **Monitoring Performance and Analysis**

#### Monitoring Performance
Regularly monitor accuracy and loss during training and testing using tools like TensorBoard.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Inside the training loop
writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
# Inside the testing loop
writer.add_scalar('Accuracy/test', accuracy, epoch)

# At the end of training
writer.close()
```

### 6. **Experimentation and Adjustment**
- Experiment and adjust hyperparameters (like learning rate, batch size).
- Evaluate the impact of adjustments on the model and choose the most suitable ones.

### Improved Code Example:

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Data Augmentation
train_transform = Compose([
    Resize([224, 224]),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor()
])
test_transform = Compose([
    Resize([224, 224]),
    ToTensor()
])

# Datasets and Dataloaders
train_dataset = torchvision.datasets.Flowers102(root='./dataset', split='train', transform=train_transform, download=True)
test_dataset = torchvision.datasets.Flowers102(root='./dataset', split='val', transform=test_transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=True)

# Model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 102)

# Loss Function and Optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# TensorBoard
writer = SummaryWriter()

# Training Function
def train():
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, targets = data
        outputs = model(images)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
    return total_loss / len(train_dataloader)

# Testing Function
def test():
    model.eval()
    total_accuracy = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, targets = data
            outputs = model(images)
            accuracy = (outputs.argmax(1) == targets).float().mean().item()
            total_accuracy += accuracy
            writer.add_scalar('Accuracy/test', accuracy, epoch * len(test_dataloader) + i)
    return total_accuracy / len(test_dataloader)

# Main Function
best_accuracy = 0.0
for epoch in range(15):
    train_loss = train()
    test_accuracy = test()
    scheduler.step()
    print(f'Epoch {epoch}: Loss = {train_loss}, Accuracy = {test_accuracy}')
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best.pth')
        best_accuracy = test_accuracy

writer.close()
```

This way, you will be able to systematically and cumulatively improve your model's performance based on continuous monitoring and analysis.




---




this link is a project on ***google colab***:
https://colab.research.google.com/drive/1yKb2VxO1c_Pdl1hLQ1XgBRHuZfinNaMu?usp=sharing

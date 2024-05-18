# Improving Accuracy in [Image-classification-via-convolution-nerual-network. "Experiment 1"](https://github.com/WhynotChen0105/Experiment-1/tree/main)

<details>
  <summary><b><font color="#0000FF">Explanation for Experiment-1</font></b></summary>
  
  ## Experiment Objective:
  The primary aim of this experiment is to achieve superior accuracy in image classification tasks using Convolutional Neural Networks (CNNs). By following a meticulously structured workflow, we endeavor to optimize the model's performance and deliver robust classification capabilities.
  
  ## Experiment Workflow:
  ### 1. Data Preparation:
  - **Dataset Collection:** Curate a labeled dataset of images, leveraging common datasets like CIFAR-10, MNIST, or crafting a custom dataset tailored to the task at hand.
  - **Data Preprocessing:** Normalize images, ensuring consistent brightness and contrast, resize them to a uniform size, and partition the dataset into training, validation, and test sets.
  
  ### 2. Model Design:
  - **Architecture Selection:** Choose an appropriate CNN architecture, ranging from simple models to sophisticated architectures like ResNet50, VGG16, or MobileNet.
  - **Layer Configuration:** Configure CNN layers, including convolutional layers for feature extraction, pooling layers for downsampling, dropout layers for regularization, and fully connected layers for classification.
  
  ### 3. Training the Model:
  - **Loss Function:** Select a suitable loss function, such as categorical cross-entropy, tailored to multi-class classification tasks.
  - **Optimizer:** Choose an optimizer like Adam, SGD, or RMSprop to minimize the loss function and update model parameters.
  - **Data Augmentation:** Apply data augmentation techniques such as rotation, flipping, and zooming to increase training data variability.
  - **Training Process:** Train the CNN on the training dataset, monitor performance on the validation set, and adjust hyperparameters as needed for optimal performance.
  
  ### 4. Model Evaluation:
  - **Performance Metrics:** Evaluate the trained model using metrics such as accuracy, precision, recall, and F1-score on the test dataset.
  - **Confusion Matrix:** Analyze the confusion matrix to understand classification performance across different classes.
  
  ### 5. Fine-Tuning and Optimization:
  - **Hyperparameter Tuning:** Experiment with different hyperparameters like learning rate, batch size, and epochs to optimize model performance.
  - **Transfer Learning:** Utilize pre-trained models and fine-tune them on the specific dataset to leverage learned features and improve classification accuracy.
  - **Regularization Techniques:** Implement dropout and batch normalization to prevent overfitting and enhance model generalization.
  
  ### 6. Deployment:
  - **Model Export:** Save the trained model in a suitable format for deployment.
  - **Inference Pipeline:** Develop an inference pipeline for real-time or batch image classification.
  - **Deployment Environment:** Deploy the model to a production environment such as a cloud service, edge device, or mobile application for practical use.
  
  ## Experiment Outcome:
  By meticulously following these steps, our experiment endeavors to achieve unparalleled accuracy in image classification tasks using CNNs, thereby paving the way for transformative advancements in the field.
</details>

---

***This project aims to improve image classification accuracy using Convolutional Neural Networks. After conducting the experiment and obtaining the results, improvements will be applied to meet the ***homework*** assignment requirements***.

After downloading and installing the trial environment requirements.
<br />
***The model will***:
- A dataset of images is loaded and transformations are applied to it.
- A ResNet-18 model is built and loaded.
- The loss function and optimizer are defined.
- The model is trained and tested over multiple epochs, and the best-performing model is saved based on accuracy.

<img src="https://github.com/2aid-dev/Image-classification-via-convolution-nerual-network./assets/42585484/2c21cbdf-c608-4fa2-b09d-275ec5d4caaa" alt="drawing" width="100%" height="600"/>

---
# Let's Improve Image Classification Accuracy: Time to Tackle the Homework!.
<sub>
You need to improve the accuary of the Image Classification task to more than 75% just by modifying the code instead of rewriting the code all. There are two things you need to submit: 1 Your code modified based on the code we provided. 2 Screenshot of the experimental results and the accuary must be more than 75%.

Hints: You can experiment with different learning rates, use the weight decay, use data augmentation and use the scheduler to change the learning rates during training the model.
</sub>


## Improving the performance of your model can be achieved through a variety of strategies. Here are some ways to enhance your model:


### Detailed Explanation of Steps and Improvements

Certainly! Let's go through the entire new code and compare it with the original code, explaining the improvements and steps taken.

### Original Code

```python
import torch
import torchvision
from torch.utils.data import DataLoader

# Build the transform for the train dataset and test dataset.
train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize([224, 224]),
     torchvision.transforms.ToTensor()])

test_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize([224, 224]), torchvision.transforms.ToTensor()])

# Build the train dataset and test dataset
train_dataset = torchvision.datasets.Flowers102(root='./dataset', split='train', transform=train_transform,
                                                download=True)
test_dataset = torchvision.datasets.Flowers102(root='./dataset', split='val', transform=test_transform,
                                               download=True)

# Build the train dataloader and test dataloader use the two datasets.
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=True)

# Build the Convolution Neural Network
# But we can build the resnet18 easily by the torchvision:
model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights)

# Build the loss function and optimizer
# For the Image Classification task, we use the cross-entropy loss function as the loss function and use the SGD(Stochastic Gradient Descent) as the optimizer.

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Train the model
def train():
    all_loss = 0.0
    n = 0
    for data in train_dataloader:
        n = n + 1
        optimizer.zero_grad()  # Set 0 into grads of optimizer
        image, target = data  # Fetch the data and target
        output = model(image)  # Forward
        loss = loss_function(output, target)  # Calculate the loss
        loss.backward()  # Backward
        optimizer.step()  # Optimizer works
        all_loss += loss.item()
        print('Train process: %.3f of this epoch, loss : %.2f ' % (n / len(train_dataloader), loss.item()))
    return all_loss / len(train_dataloader)  # return the loss

# Test the model
def test():
    model.eval()  # set the model into the evaluation mode, stopping Backward.
    all_acc = 0.0
    n = 0
    for data in test_dataloader:
        n = n + 1
        image, target = data  # Fetch the data
        output = model(image)  # Forward
        print('Test process: %.2f of this epoch' % (n / len(test_dataloader)))
        all_acc += torch.eq(torch.argmax(output, dim=-1), target).float().mean()  # Partial accuary
    model.train()  # set the model into training mode
    return all_acc / len(test_dataloader)

def main():
    best_acc = 0.0
    for i in range(15):  # train for 15 epochs
        # best accuary
        loss = train()
        acc = test()
        print(f"epoch: {i}, loss: {loss}, accuary: {acc}")
        if acc > best_acc:
            torch.save(model, 'best.pth')  # save the best model
            best_acc = acc

if __name__ == '__main__':
    main()
```

### Improved Code

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms and datasets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.Flowers102(root='./dataset', split='train', transform=train_transform, download=True)
test_dataset = torchvision.datasets.Flowers102(root='./dataset', split='val', transform=test_transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

# Initialize the model
model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 102)  # Change the last layer for 102 classes
model = model.to(device)

# Define loss function, optimizer, and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # StepLR scheduler

# Train the model
def train():
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU/CPU)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(train_dataloader), correct / total

# Test the model
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device (GPU/CPU)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return correct / total

def main():
    best_acc = 0.0
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10
    epoch = 0
    accuracy_threshold = 0.75  # Step 1: Define the accuracy threshold

    while True:
        start_time = time.time()
        train_loss, train_acc = train()
        test_acc = test()
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch [{epoch+1}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_duration:.2f}s")
        scheduler.step()

        if test_acc > best_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            best_acc = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check if accuracy exceeds the threshold
        if test_acc >= accuracy_threshold:
            print(f"Accuracy exceeds {accuracy_threshold * 100}%!")
            break

        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Early stopping due to no improvement in validation accuracy")
            break

        epoch += 1

if __name__ == '__main__':
    main()
```

### Detailed Explanation of Steps and Improvements

1. **Setting Random Seed**:
   - **Original Code**: Did not set a random seed for reproducibility.
   - **Improved Code**:
     ```python
     torch.manual_seed(42)
     ```

2. **Transformations and Data Augmentation**:
   - **Original Code**: Basic transformations without data augmentation.
     ```python
     train_transform = torchvision.transforms.Compose(
         [torchvision.transforms.Resize([224, 224]),
          torchvision.transforms.ToTensor()])
     test_transform = torchvision.transforms.Compose(
         [torchvision.transforms.Resize([224, 224]), torchvision.transforms.ToTensor()])
     ```
   - **Improved Code**: Added advanced data augmentation techniques for better generalization.
     ```python
     train_transform = transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
         transforms.RandomRotation(20),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     test_transform = transforms.Compose([
         transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     ```

3. **Batch Size Adjustment**:
   - **Original Code**





## Results:
![image](https://github.com/2aid-dev/Image-classification-via-convolution-nerual-network./assets/42585484/64970023-5e44-4fa4-83a9-dd3fe5b9261a)
### After 15 times of training the model and testing it, we did not get the desired result. So let's increase the number of times we train and test

---


## Google Colab Project Links
[Original Version (Experiment 1)](https://colab.research.google.com/drive/1NiNzSLBAbhRZ-5vu2cbATx3Pj3lDv_aw?usp=sharing)

[After Improving Accuracy (Experiment 1)](https://colab.research.google.com/drive/1yKb2VxO1c_Pdl1hLQ1XgBRHuZfinNaMu?usp=sharing)



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
# So let's do the homework ().

<sub><sup>combining the two tags</sup></sub>

<span style="font-size:0.5em;">

You need to improve the accuary of the Image Classification task to more than 75% just by modifying the code instead of rewriting the code all. There are two things you need to submit: 1 Your code modified based on the code we provided. 2 Screenshot of the experimental results and the accuary must be more than 75%.

Hints: You can experiment with different learning rates, use the weight decay, use data augmentation and use the scheduler to change the learning rates during training the model.

</span>





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

This way, you will be able to systematically and cumulatively improve your model's performance based on continuous monitoring and analysis.




---



## Google Colab Project
[Link to Google Colab Project](https://colab.research.google.com/drive/1yKb2VxO1c_Pdl1hLQ1XgBRHuZfinNaMu?usp=sharing)


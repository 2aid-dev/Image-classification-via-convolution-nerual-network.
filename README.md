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

### Detailed Explanation of Steps and Improvements

1. **Setting Random Seed**:
   - Did not set a random seed for reproducibility.
     
     ```python
     torch.manual_seed(42)
     ```

3. **Transformations and Data Augmentation**:
   - **Original Code**: Basic transformations without data augmentation.
   
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

4. **Batch Size Adjustment**:
   - **Original Code**: Batch size set to 128.
   - **Improved Code**: Reduced batch size to 64 to potentially improve model convergence and handle data augmentation.
     ```python
     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
     ```

5. **Using Pre-trained Model with Fine-Tuning**:
   - **Original Code**: Loaded the pre-trained ResNet-18 without specifying custom output classes.
   - **Improved Code**: Modified the final layer to match the number of output classes (102).
     ```python
     model = resnet18(pretrained=True)
     num_ftrs = model.fc.in_features
     model.fc = torch.nn.Linear(num_ftrs, 102)  # Change the last layer for 102 classes
     ```

6. **Optimizer and Learning Rate Scheduler**:
   - **Original Code**: Used SGD optimizer without a learning rate scheduler.
    
   - **Improved Code**: Switched to Adam optimizer for potentially better performance and added a learning rate scheduler.
     ```python
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
     scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
     ```

7. **Device Handling (GPU/CPU)**:
   - **Original Code**: Did not specify device usage.
   - **Improved Code**: Added device handling to utilize GPU if available.
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model = model.to(device)
     ```

8. **Training and Testing Functions**:
   - **Original Code**: Basic implementation without device handling or accuracy calculation.
     
   - **Improved Code**: Enhanced with device handling, accuracy calculation, and logging.
     ```python
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
     ```

9. **Main Function with Early Stopping and Progress Logging**:
   _ **Original Code**: Basic loop for 15 epochs without early stopping or progress logging.
   _ **Improved Code**: Added early stopping, progress logging, and accuracy threshold check.
     ```python
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

### Summary of Improvements:
1. **Setting a Random Seed**: Ensures reproducibility of results.
2. **Data Augmentation**: Enhances training with diverse transformations, improving model generalization.
3. **Batch Size Adjustment**: Optimizes training and handles data augmentation better.
4. **Pre-trained Model Fine-Tuning**: Adjusts the final layer to match the specific number of classes (102).
5. **Optimizer and Learning Rate Scheduler**: Uses Adam optimizer for better performance and introduces a learning rate scheduler.
6. **Device Handling**: Utilizes GPU for faster computation if available.
7. **Enhanced Training and Testing Functions**: Includes device handling, accuracy calculation, and logging.
8. **Main Function Enhancements**: Introduces early stopping, progress logging, and an accuracy threshold check.

These changes improve the training process, model performance, and provide better insights during model training and evaluation.























## Results:
![image](https://github.com/2aid-dev/Image-classification-via-convolution-nerual-network./assets/42585484/64970023-5e44-4fa4-83a9-dd3fe5b9261a)
### After 15 times of training the model and testing it, we did not get the desired result. So let's increase the number of times we train and test

---


## Google Colab Project Links
[Original Version (Experiment 1)](https://colab.research.google.com/drive/1NiNzSLBAbhRZ-5vu2cbATx3Pj3lDv_aw?usp=sharing)

[After Improving Accuracy (Experiment 1)](https://colab.research.google.com/drive/1yKb2VxO1c_Pdl1hLQ1XgBRHuZfinNaMu?usp=sharing)



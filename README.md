# CNN vs. GCNN for Rotated MNIST Classification
Sean Yu, Ziqing Lin
This repository contains the results and analysis of experiments comparing Convolutional Neural Networks (CNNs) with Graph Convolutional Neural Networks (GCNNs) in the context of classifying rotated MNIST digits.
## Overview
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## Experiment Setup
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
- **Parameter Choices:** Identical model parameters for both CNN and GCNN. We used the same learning rate, batch size, and max epoch.

- **Degree of Rotation:** We systematically varied the degree of rotation for training data from 0 to 360 degrees to assess model adaptability to rotated images.


## Experiment Results

- **Accuracy Comparison between CNN and GCNN:** GCNN consistently outperformed CNN in terms of accuracy when classifying rotated MNIST digits, suggesting that it is better at handling images with varying orientations.

The following plots compare the test accuracy of a Convolutional Neural Network (CNN) and a Graph Convolutional Neural Network (GCNN) over 10 epochs of the full MNIST dataset (60k instances). It is found that the CNN's ability to generalize and predict accurately on the test set increase as the input data's rotational degree increase, while there is no clear trend of how GCNN's performance varies with the degree of rotation. Moreover, CNN test accuracies generally show a steady increase with each epoch,  suggesting the model is learning and improving its predictions over time, while GCNN test accuracies display more fluctuation across epochs. 
![cnn-test-accuracy-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/4d30b121-7331-4320-a1ff-37ebbec80e20)
![gcnn-test-accuracy-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/5afd8234-44fe-4aea-ae95-06f223db8c1c)

- **Accuracy Comparison between CNN and GCNN:** GCNN consistently displays lower loss compared to CNN when classifying rotated MNIST digits. Similar to the results of accuracy, the test loss for CNN decreases as epochs increase, which is the opposite trend as the accuracy. This is expected since loss can be viewed as the deviation between the true values and the predicted values, the lower the loss, the smaller the error, thus higher the accuracy. In addition, it is also found that there is no clear trend of how GCNN's performance varies with the degree of rotation.

Both the accuracy curve and the loss curve have shown a clear trend between CNN and the degree of rotation and a lack of trend for the GCNN model. 
![cnn-test-loss-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/1ddb34aa-8d74-4cb2-9aab-ff01cf0f8bc8)
![gcnn-test-loss-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/ca0f8c32-6bc7-4040-be5c-6d09bcf247da)


- **Cost-Effectiveness of GCNN:**  By comparing the test accuracy of a Graph Convolutional Neural Network (GCNN) trained with 10k, 30k, and 60k data instances over 10 epochs, it is observed that there is a clear correlation between dataset size and model performance. The larger the datasets the faster and higher the accuracy across all degrees of rotation.

For 10k dataset, GCNN model initially shows a steep learning curve from epoch 1 to epoch 4, after epoch 4, the test accuracy starts to plateau, with only slight improvements or variations up to epoch 10.
For 30k dataset, shows a steeper learning curve, suggesting that the model learns more effectively than the 10k dataset. After epoch 4, GCNN reaches plateau with most degrees reaching over 80% accuracy.
For 60k dataset,  GCNN model shows the steepest learning curve, and reaches the highest accuracy with most lines surpassing 90% accuracy.
![gcnn-test-loss-10k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/f8e990f5-28cf-464f-8ce3-2d5355df4e1f)
![gcnn-test-loss-30k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/c18aad63-7416-41ef-a179-3f5800157c2a)
![gcnn-test-accuracy-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/011bdd00-28b0-4547-b970-27f032456ff0)


## Discussion
- **Stability vs. Fluctuation:** CNN's test accuracy remained stable across different degrees of rotation, while GCNN's accuracy showed more fluctuation, indicating potential flexibility but occasional performance variations.

- **Cost-Effectiveness:** Both models were evaluated on subsets of the MNIST dataset:
  - With 10,000 samples, both models displayed varying accuracies, with some lines reaching or exceeding 80%.
  - Using 30,000 samples, the accuracies generally improved, with a majority falling between 80% and 90%.
  - With the full 60,000-sample dataset, both models excelled, with most lines surpassing 90% accuracy. GCNN proved cost-effective with reduced datasets.
    
Our findings suggest that GCNN holds an advantage over CNN in handling rotated images, potentially impacting various computer vision tasks. The observed fluctuations in GCNN's accuracy call for further investigation and fine-tuning strategies.

In terms of cost-effectiveness, GCNN's ability to maintain high accuracy with reduced datasets can reduce data acquisition costs and computational requirements.

Future work may involve comparing results with existing state-of-the-art approaches, mitigating performance fluctuations in GCNN, and optimizing the model's architecture for rotated image tasks.




# CNN vs. GCNN for Rotated MNIST Classification
Sean Yu, Ziqing Lin

This repository contains the results and analysis of experiments comparing Convolutional Neural Networks (CNNs) with Graph Convolutional Neural Networks (GCNNs) in the context of classifying rotated MNIST digits.

## Overview

This empirical study explore the extent to which equivariant neural networks surpass traditional neural network architectures in terms of performance and generalization across MNIST datasets. Our goal is evaluating the impact of data augmentation on the efficacy of equivariant models and analyzing how the quantity of data influences their overall performance.

## Experiment Setup

- **Parameter Choices:** Identical model parameters for both CNN and GCNN. We used the same learning rate, batch size, and max epoch.

- **Degree of Rotation:** Varied the degree of rotation for training data in the range [0, 30], [0, 60], ..., [0, 360], and evaluated both CNN and GCNN models on an augmented MNIST test set with images randomly rotated between [0, 360].



## Experiment Results

- **Accuracy Comparison between CNN and GCNN:** GCNN consistently outperformed CNN in terms of accuracy when classifying rotated MNIST digits, suggesting that it is better at handling images with varying orientations.

The following plots compare the test accuracy of a Convolutional Neural Network (CNN) and a Graph Convolutional Neural Network (GCNN) over 10 epochs of the full MNIST dataset (60k instances). It is found that the CNN's ability to generalize and predict accurately on the test set increase as the input data's rotational degree increase, while there is no clear trend of how GCNN's performance varies with the degree of rotation. Moreover, CNN test accuracies generally show a steady increase with each epoch,  suggesting the model is learning and improving its predictions over time, while GCNN test accuracies display more fluctuation across epochs. 
![cnn-test-accuracy-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/4d30b121-7331-4320-a1ff-37ebbec80e20)
![gcnn-test-accuracy-60k](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/5afd8234-44fe-4aea-ae95-06f223db8c1c)

The following figure under varying degrees of rotation (30, 180, and 360 degrees) is plotted to further compare the accuracy between CNN and GCNN. 

At 30 degrees of rotation, the accuracy of GCNN is significantly higher than that of CNN, indicating that GCNN has better rotational invariance compared to CNN. The gap between the two models is large, with GCNN having accuracy above 0.8, while CNN accuracy is  above 0.4.
![30](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/0ba9c372-6900-4510-9af6-030e18ce0161)

At 180 degrees of rotation, both models experience an increase in accuracy compared to the 30 degrees scenario, but the increase is more prominent for the CNN. The gap between the two models starts to shrink, indicating that as the degree of rotation increases, with GCNN having accuracy above 0.8, while CNN accuracy is  above 0.7 , CNN's performance increase at a faster rate than GCNN. GCNN also shows a notable dip in performance around the midpoint of the epochs, but it recovers towards the end.
![180](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/459ce076-555f-498b-8ede-827a4bb40381)

At 360 degrees of rotation, the performance gap between CNN and GCNN further narrows with GCNN having overall accuracy above 0.9, while CNN accuracy is  above 0.8 . GCNN still have better overall accuracy than CNN, but at the beginning, CNN displays higher accuracy than  the accuracy of GCNN. 
![360](https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/9e8762ca-0322-4622-ba67-3baa8e16d5bc)


<!-- 
<div style="display: flex;">
    <img src="https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/ab34b3c1-435c-46e2-9bcf-3a4794cabbcf" alt="Image 1" width="33%">
    <img src="https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/0bd1afff-ea04-46f2-940e-ca84dfb0f96c" alt="Image 2" width="33%">
    <img src="https://github.com/janeyziqinglin/equivariant_neural_network/assets/105125897/3bbbbf9e-c078-47df-a97f-8a25789a135c" alt="Image 3" width="33%">
</div>
-->




- **Loss Comparison between CNN and GCNN:** GCNN consistently displays lower loss compared to CNN when classifying rotated MNIST digits. Similar to the results of accuracy, the test loss for CNN decreases as epochs increase, which is the opposite trend as the accuracy. This is expected since loss can be viewed as the deviation between the true values and the predicted values, the lower the loss, the smaller the error, thus higher the accuracy. In addition, it is also found that there is no clear trend of how GCNN's performance varies with the degree of rotation.

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
- **Accuracy Comparison**  The results suggest that while GCNN generally outperforms CNN in terms of accuracy, especially with lower degrees of rotation, the performance advantage of GCNN diminishes as the degree of rotation increases. This is because of the design of GCNN as discussed in introduction, allowing it to learn and generalize better than CNN, leading to higher test accuracy. On the other hand, CNN's performance, while lower at low degree of rotation, shows more statbility over epoch. CNN's test accuracy remained stable across different epoch, while GCNN's accuracy showed more fluctuation, indicating potential flexibility but occasional performance variations.

- **Cost-Effectiveness:** Both models were evaluated on subsets of the MNIST dataset:
  - With 10,000 samples, both models displayed varying accuracies, with some lines reaching or exceeding 80%.
  - Using 30,000 samples, the accuracies generally improved, with a majority falling between 80% and 90%.
  - With the full 60,000-sample dataset, both models excelled, with most lines surpassing 90% accuracy. GCNN proved cost-effective with reduced datasets.

## Conclusion
To summarize, GCNN has a significant advantage over CNN at lower degrees of rotation due to its ability to leverage the geometric structure of the data. However, as the degree of rotation increases in the training data, the advantage of GCNN diminishes. In terms of cost-effectiveness, GCNN's ability to maintain high accuracy with reduced datasets can reduce data acquisition costs and computational requirements. Future work can be done by  mitigating performance fluctuations in GCNN by hyperparameter tuning, and optimizing the model's architecture for rotated image tasks.
Last but not least, this study have shed light to model selection in real world application. When dealing with complicated medical images rather than simple MNIST data, it is likely that GCNNs might be more adept due to their ability to exploit relational or geometric structure, additionally, GCNNs might require less data to achieve high performance. However, CNNs might perform equally well by preprocessing the training data to include the expected variations in the test data, e.g. rotations.

## References





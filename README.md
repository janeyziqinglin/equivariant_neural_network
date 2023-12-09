# CNN vs. GCNN for Rotated MNIST Classification

This repository contains the results and analysis of experiments comparing Convolutional Neural Networks (CNNs) with Graph Convolutional Neural Networks (GCNNs) in the context of classifying rotated MNIST digits.



## Experiment Setup

- **Parameter Choices:** Identical model parameters for both CNN and GCNN ensured a fair comparison. We used the same learning rate, batch size, and optimizer settings.

- **Degree of Rotation:** We systematically varied the degree of rotation for training data from 0 to 360 degrees to assess model adaptability to rotated images.


## Experiment Results

- **Accuracy Comparison:** GCNN consistently outperformed CNN in terms of accuracy when classifying rotated MNIST digits, suggesting its superiority in handling images with varying orientations.

- **Stability vs. Fluctuation:** CNN's test accuracy remained stable across different degrees of rotation, while GCNN's accuracy showed more fluctuation, indicating potential flexibility but occasional performance variations.

- **Cost-Effectiveness:** Both models were evaluated on subsets of the MNIST dataset:
  - With 10,000 samples, both models displayed varying accuracies, with some lines reaching or exceeding 80%.
  - Using 30,000 samples, the accuracies generally improved, with a majority falling between 80% and 90%.
  - With the full 60,000-sample dataset, both models excelled, with most lines surpassing 90% accuracy. GCNN proved cost-effective with reduced datasets.


## Discussion

Our findings suggest that GCNN holds an advantage over CNN in handling rotated images, potentially impacting various computer vision tasks. The observed fluctuations in GCNN's accuracy call for further investigation and fine-tuning strategies.

In terms of cost-effectiveness, GCNN's ability to maintain high accuracy with reduced datasets can reduce data acquisition costs and computational requirements.

Future work may involve comparing results with existing state-of-the-art approaches, mitigating performance fluctuations in GCNN, and optimizing the model's architecture for rotated image tasks.




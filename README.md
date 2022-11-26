# Machine Learning 

This repository includes handouts and codes I used for delivering the Machine Learning module at the Department of Physics of the University of Liverpool.

## Author

Prof. Costas Andreopoulos, **FHEA**  < constantinos.andreopoulos \at cern.ch >

<pre>
 University of Liverpool          |  U.K. Research & Innovation (UKRI)
 Faculty of Science & Engineering |  Science & Technology Facilities Council (STFC)
 School of Physical Sciences      |  Rutherford Appleton Laboratory 
 Department of Physics            |  Particle Physics Department
 Oliver Lodge Lab 316             |  Harwell Oxford Campus, R1 2.89
 Liverpool L69 7ZE, UK            |  Oxfordshire OX11 0QX, UK          
 tel: +44-(0)1517-943201          |  tel: +44-(0)1235-445091 
</pre>


## Aims of the module
- To introduce the fundamental concepts of machine learning.
- To develop the ability of students to address common real-world problems using one of the leading open source frameworks for machine learning.

## Syllabus
- Training neural networks, backpropagation
- Supervised and unsupervised learning algorithms
- Stochastic Gradient Descent (SGD)
- Batch normalization
- Convolutional Neural Networks (CNN)
- Classification and semantic segmentation using CNNs
- Generative Adversarial Networks (GANs)
- Case studies from science and industry
- Least-squares regression and classification.
- Widrow-Hoff learning rule; Adaptive linear neuron (Adaline); Delta rule; Fisher discriminant
- Logistic regression
- Support vector machines

## Structure

### Lecture  - Introduction ( hr)

- Precursors of artificial intelligence
- History of artificial intelligence
- Human vs computer learning
- Different learning paradigms: Supervised, unsupervised and reinforcement learning
- Artificial intelligence, machine learning and deep learning
- Machine learning tasks
- A simple practical example: Linear regression.
- Biologically inspired methods of computer learning
- Basic architecture of neural networks
- Fundamental concepts

### Lecture  - Shallow Neural Networks, part I ( hr)

- Single-layer networks: The perceptron.
- Loss functions.
- Heuristic optimization of the original Mark I perceptron.
- Stochastic gradient-descent method.
- Percepton criterion.
- Activation functions and their properties.
- Variants of the perceptron and connection with other regression and classification models.
- Least-squares regression and classification. Widrow-Hoff learning rule.
- Closed form solutions of least-squares regression.
- Logistic regregression.
- Support vector machines.

### Lecture  - Shallow Neural Networks, part II ( hr)

- The multiclass perceptron.
- Multiclass (Weston-Watkins) support vector machines.
- Multinomial logistic regression (Softmax classifier).
 
### Lecture  - Introduction to Deep Networks

- Multi-layer networks: Deep feedforward networks, or multilayer perceptons
- Simple example: Learning XOR 

### Lecture - Learning algorithms and backpropagation ( hr)
- Gradient-based optimization
- Stochastic gradient descent
- Polyak momentum
- Nesterov momentum
- Adaptive learning rates
- Second order methods
    - Beyond the gradient: Jacobian and Hessian matrices
    - Newton's method
    - Conjugate gradient
    - Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm
- Batch normalization
- Supervised pretraining / greedy algorithms
- Training neural networks with backpropagation

### Lecture  - Practical issues in neural network training ( hr)

- Capacity, overfitting and underfitting
- Vanishing and exploding gradient problems
  - Leaky ReLU and maxout 
- Difficulties in convergence
- Local optima
- Bias-variance trade-off
- Regularization methods for deep learning
  - Norm penalties: L2 and L1 regularization
  - Dataset augmentation
  - Noise robustness
  - Multitask learning
  - Early stopping
  - Parameter sharing
  - Ensemble methods: Bagging, subsampling and dropout
  - Adversarial Training

### Lecture  - Convolutional Neural Networks (1 hr)
  - Historical perspective and neuroscientific basis
  - Convolution and cross-correlation
  - Motivation: Sparse interactions, parameter sharing and equivariant representations
  - Basic structure of convolutional neural networks
     - Padding
     - Strides
     - ReLU layer
     - Pooling
     - Fully connected layers
     - 
  - Training a convolutional neural network
  - Typical convolutional architectures
     - AlexNet
     - ZFNet
     - VGG
     - GoogLeNet
     - ResNet
  - The effects of depth
  - Applications


### Lecture  - Recurrent and recursive networks ( hr)

### Lecture  - Autoencoders ( hr)

### Lecture  - Deep Reinforcement Learning ( hr)

### Lecture  - Deep Generative Models (1 hr)


### Workshop 1 (2 hrs)


### Workshop 2 (2 hrs)


### Workshop 3 (2 hrs)


### Workshop 4 (2 hrs)


### Workshop 5 (2 hrs)


### Project (10 hrs)


## Reading list

### Key textbooks

- Ian Goodfellow, Yoshua Bengio and Aaron Courville, Deep Learning, MIT Press (2016)
- Charu Aggarwal, Neutral Networks and Deep Learning - A Textbook, Springer (2018)
 
### Other useful reading

- Aurélien Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow - Concepts, Tools, and Techniques to Build Intelligent System, O'Reilly, 2019

### PyTorch documentation and tutorials

- Official PyTorch documentation, https://pytorch.org/docs/stable/
- Official PyTorch tutorials, https://pytorch.org/tutorials/
- PyTorch tutorial by Giles Strong, https://github.com/GilesStrong/pytorch_tutorial_2022

### Relevant research articles

- ImageNet Classification with Deep Convolutional Neural Networks, https://doi.org/10.1145/3065386
- GoogleNet, arXiv:1409.4842
- R-CNN, arXiv:1311.2524
- Fast R-CNN, arXiv:1504.08083
- Faster R-CNN, arXiv:1506.01497
- Fully Convolutional Neural Nets for Semantic Segmentation, arXiv:1605.06211
- Mask R-CNN, arXiv:1703.06870
- Multiview CNN, arXiv:1505.00880
- Generative Adversarial Nets, arXiv:1406.2661
- Deep Convolutional Generative Adversarial Nets, arXiv:1511.06434

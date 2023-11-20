# Machine Learning 

This repository includes handouts and codes I used for delivering the Machine Learning module at the Department of Physics of the University of Liverpool.

## Author

**Professor Costas Andreopoulos**, *FHEA*  

Department of Physics, **University of Liverpool**  <br />
Oliver Lodge Lab 316, Liverpool L69 7ZE, UK 

e-mail: constantinos.andreopoulos \at cern.ch

tel: +44-(0)1517-943201 (office), +44-(0)7540-847333 (mobile)

## Overview

Machine Learning (ML) is the study and development of techniques empowering machines to gain knowledge through experience, deduction, or reasoning, and to make intelligent decisions. Since pioneering developments in 1950’s, the field of ML, and, more broadly, of Artificial Intelligence (AI), has ebbed and flowed. During the last 10 years, increased computational power, availability of very large datasets, architectural and algorithmic innovations and development of transfer learning techniques, and interdisciplinary collaboration have powered a revolution in ML. Modern deep neural architectures are transforming industries by improving efficiency, automation, and ability to enhance insights from very complex datasets. This is a very active field of research, with a host of awe-inspiring and disrupting applications developed at an astonishing rate. The module will develop an understanding of the general principles of ML, as well as an appreciation of practical issues in the training of ML models, and a knowledge of diagnostic tools and improved algorithms available to practitioners. Special emphasis will be given in neural architectures, as a unified framework that can encompass and emulate several other traditional ML approaches. Several shallow and deep neural architectures will be studied in detail, both at a more abstract mathematical level and at a practical level, using Python and one of most popular open-source deep learning frameworks used in industry and research (pyTorch). Ethical considerations in ML will be introduced. Strong emphasis shall be given in case studies from several areas of physical sciences, medical imaging and technology.

## Aims of the module

To develop a solid understanding of fundamental concepts in Machine Learning (ML) for supervised, unsupervised and reinforcement learning; to introduce students to a broad range of shallow and deep neural architectures, and to demonstrate connections between neural architectures and more traditional ML models; to teach students how to formulate real-world problems as ML problems; to develop an understanding of general optimization and well as of the crucial differences between general and ML optimization, and to draw attention to a broad range of practical issues in ML model training; to teach students how to assess model performance and diagnose issues, and to introduce several tricks and improved optimisation algorithms available to ML practitioners; to develop practical skills in the implementation of ML models using a programming language like Python and popular open-source ML libraries such as pyTorch;  to develop experience in the use of common modern ML architectures, such as convolution neural networks (CNNs) or recurrent neural networks (RNNs); to illustrate a broad range of practical applications, with special emphasis in the areas of physical sciences, medical imaging and technology; to introduce problem of bias in Machine Learning and demonstrate an understanding of ethical and responsible practices.

## Learning outcomes 

LO1. Demonstrate a solid understanding of fundamental concepts of Machine Learning.

LO2. Demonstrate good knowledge of key Machine Learning algorithms and models.

LO3. Demonstrate an understanding of practical issues in the training of Machine Learning models, and a knowledge of the broad range of tricks and approaches available to Machine Learning practitioners.

LO4. Apply key Machine Learning algorithms to a variety of simple problems and demonstrate an ability to evaluate and optimize model performance.

LO5. Appreciate the problem of bias in Machine Learning and demonstrate an understanding of ethical and responsible practices.

## Skills

S1. Problem solving skills.

S2. Practical implementation skills: Apply Machine Learning techniques on a variety of problems, using a popular open-source deep learning frameworks used both industry and in research environments (eg. pyTorch developed by Meta AI).

S3. Collaboration and teamwork: Collaborate effectively in team projects applying machine learning techniques on a variety of problems.

S4. Research and literature review skills: Performs literature reviews and critical assessment of research papers in Machine Learning.

S5. Effective communication: Communicate complex Machine Learning concepts in discussion sessions and workshops. Communicate technical information and results from Machine Learning projects. 

S6. Adaptability to emerging technologies and lifelong learning: Develop a mindset of continuous learning in the rapidly evolving field of Machine Learning, 

## HECoS subjects

-	100992 - Machine Learning (60%)
-	100358 - Applied Computing (20%)
-	101030 - Applied Statistics (20%)

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

### Part 01 - Introduction to Machine Learning (2 hrs)

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

### Part 02 - Shallow Neural Networks (3 hrs)

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
- The multiclass perceptron.
- Multiclass (Weston-Watkins) support vector machines.
- Multinomial logistic regression (Softmax classifier).
 
### Part 03 - Introduction to Deep Networks (2 hrs)

- Multi-layer networks: Deep feedforward networks, or multilayer perceptons
- Simple example: Learning XOR 

### Part 04 - Learning algorithms and backpropagation (2 hrs)

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

### Part 05 - Practical issues in neural network training (2 hrs)

- Capacity, overfitting and underfitting
- Vanishing and exploding gradient problems
  - Leaky ReLU and maxout 
- Difficulties in convergence
- Local optima
- Bias-variance trade-off
- Regularisation methods for deep learning
  - Norm penalties: L2 and L1 regularisation
  - Dataset augmentation
  - Noise robustness
  - Multitask learning
  - Early stopping
  - Parameter sharing
  - Ensemble methods: Bagging, subsampling and dropout
  - Adversarial Training

### Part 06 - Convolutional Neural Networks (2 hrs)

  - Historical perspective and neuroscientific basis
  - Convolution and cross-correlation
  - Motivation: Sparse interactions, parameter sharing and equivariant representations
  - Basic structure of convolutional neural networks
     - Padding
     - Strides
     - ReLU layer
     - Pooling
     - Fully connected layers
  - Training a convolutional neural network
  - Typical convolutional architectures
     - AlexNet
     - ZFNet
     - VGG
     - GoogLeNet
     - ResNet
  - The effects of depth
  - Applications


### Part 07 - Recurrent Neural Networks (2 hrs)

  - Data types with sequential dependencies amongst attributes: Time series, text and biological data
  - Types of sequence-centric applications and challenges
  - Basic structure of Recurrent Neural Networks (RNN)
  - Training RNNs: (Truncated) Backpropagation Through Time (BPTT)
  - Bidirectional RNNs
  - Multilayer RNNs
  - Practical issues in training RNNs
  - Echo-State Networks and Liquid-State Machines
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Units (GRUs)
  - RNN applications

### Part 08 - Autoencoders (2 hrs)

### Part 09 - Deep Reinforcement Learning (2 hrs)

### Part 10 - Deep Generative Models (2 hrs)

### Part 11 - Case Studies in Science and Technology (2 hrs)
  - Event and Particle Identification in High Energy Physics
  - Generative Models for Fast Simulations 
  - Denoising Raw Data
  - Vertex and Track Reconstruction in High Energy Physics
  - Image Recognition in Astronomy
  - Real-Time Accelerator Control
  -
    
  
### Workshop 1 (2 hrs) - Tools and Platforms for Deep Learning, Introduction to PyTorch (2 hrs)


### Workshop 2 (2 hrs) -


### Workshop 3 (2 hrs) -


### Workshop 4 (2 hrs) -


### Workshop 5 (2 hrs) -


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

### Other useful resources on the web

- A Living Review of Machine Learning for Particle and Nuclear Physics, https://github.com/iml-wg/HEPML-LivingReview

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

# Machine Learning 

This repository includes handouts and codes I use for delivering the Machine Learning module at the Department of Physics of the University of Liverpool. This is a 15-credit FHEQ Level 7 module, currently **under preparation**.

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

## Outline syllabus

Artificial Intelligence (AI) and precursors of AI; Types of AI: Type 1 (Artificial Narrow Intelligence, Artificial General Intelligence, Artificial Superintelligence) and Type 2 (Reactive AI, Limited memory AI, Theory of mind AI, Self-awareness AI). 
Developments leading to the birth of AI: Theory of computation, information theory, control and communication theory, and biological inspirations from the study of the mammal brain. The Turing Test.

Brief history of AI: From the development of an artificial neuron, the Stochastic Neural Analog Reinforcement Calculator (SNARC), the Logic Theorist and the Birth of AI at the Dartmouth conference (1956) to the AI revolution starting with ImageNet Large Scale Visual Recognition Challenge in 2012. Recent evolution of AI system capabilities and survey of present landscape. 

Human vs computer learning.

Learning paradigms: Supervised, unsupervised and reinforcement learning.

Typical learning tasks: Classification; Regression; Transcription; Machine Translation; Anomaly Detection; Synthesis and Sampling; Denoising; Density Estimation.
Fundamentals of Machine Learning (ML).

Motivating the development of Deep Learning (DL); Impact of large datasets and network size / depth in network performance; Representation learning; Transfer learning.

Computational considerations; Graphical and Tensor Processing Units.

Relationship between AI, ML and DL. 

Objective function; Cost function; Loss function; Risk and empirical risk; Surrogate cost functions; Activation functions and their properties. 

Loss functions: Mean Squared Error (MSE); Mean Absolute Error (MAE); Binary Cross-Entropy Loss; Categorical Cross-Entropy Loss; 0-1 Loss.

Activation functions: Sigmoid (logistic) activation; hyperbolic tangent activation; Rectified Linear Unit (ReLU); Leaky ReLU; Parametric ReLU; Exponential Linear Unit; Softmax activation.

Estimators; Bias; Variance; Bias-Variance trade-off; Maximum Likelihood Estimation.

Single-layer networks: The perceptron. 
Historical origins: McCulloch-Pitts neuron and Rosenblat’s Mark 1 perceptron.
Heuristic optimisation of the original Mark 1 perceptron. 

Formulation of the perceptron as a typical ML problem; Derivation of the perceptron criterion.

Perceptron variants and connections with other ML models: Least-squares regression and classification, Widrow-Hoff learning rule, Logistic regression, Support Vector Machine (SVM).

The multiclass perceptron.
Multiclass (Weston-Watkins) SVM; Multinominal logistic regression (Softmax classifier).

Closed-form solutions of least-squares regression.

Using a perceptron to learn the XOR function: Limitations of linear models.

Multilayer networks: Deep feedforward networks, or multilayer perceptrons.

Introduction to Automatic Differentiation.
Forward and Backward Modes of Automatic Differentiation. Backpropagation.

Basics of general gradient-based optimisation. Gradient and Jacobian matrix; Beyond the gradient: The Hessian matrix and its properties. Function extremization; Critical points; Local and global minima and maxima; Saddle points; The second-derivative test in multiple dimensions. Condition number; Ill conditioning; Hyperparameters; Learning rate. 

Optimisation in Machine Learning: Similarities and differences with general optimisation. 

Mini-Batch Gradient Descent; Stochastic Gradient Descent (SGD);

Difficulties in convergence; The vanishing and exploding gradients problem.
Improved optimisation strategies: Learning rate decay; Momentum-based methods and Nesterov momentum; Adaptive learning rate algorithms (Delta-Bar-Delta, AdaGrad, RMSProp, RMSPrip with momentum, Adam); 2nd order methods (Newton methods and approximate methods: Conjugate gradients algorithm; Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm; Limited Memory BFGS algorithm (L-BFGS)).

Generalisation; Neural network capacity; Overfitting and underfitting; No free lunch theorem.

Regularisation techniques: L2 and L1 norm penalties; Dataset augmentation; Noise robustness Multitask learning; Early stopping; Parameter sharing; Ensemble methods (Bagging, subsampling and dropout); Adversarial training.

The Convolutional Neural Network (CNNs) and motivations for convolutional architectures: Sparse connectivity; Parameter sharing, Equivariant representations. 
Biological inspiration for CNNs: Hubel and Wiesel experiments; the visual cortex; Brodmann areas 17-19; CNN precursors: The Neocognitron.

Convolution and cross-correlation operations. 
Basic structure of CNNs: Convolution, pooling and ReLU activation layers; Fully connected layers; Feature maps and feature detectors (filters or kernels); Padding and strides; Receptive field of convolutional networks; Types of CNN pooling (or subsampling) layers: Max, average, sum and stochastic pooling. 

Modern convolutional network architectures: LeNet-5, AlexNet, GoogLeNet, ResNet.

Region-based CNN (R-CNN) and its variations (Fast/Faster R-CNN; Mask R-CNN).

Survey of CNN applications; Detailed examples for object detection and object localization.

Specialised networks for sequential data: The Recurrent Neural Network (RNN). 

Basic architecture of RNNs. Training RNNs: Backpropagation Through Time (BPTT).

Bidirectional RNNs; Multiplayer RNNs.

Echo-State Networks; Long Short-Term Memory (LSTM); Gated Recurrent Units (GRU).

Survey of RNN applications; Detailed examples for Natural Language Modelling. 

Large Language Models; Transformer architectures; Self-attention mechanisms.
Notable Large Language Models: Generative Pre-trained Transformer 3 and 4.

Advanced topics: Autoencoders; Variational Autoencoders; Deep Reinforcement Learning; Generative Adversarial Networks (GANs).

Ethical Considerations and Bias in Deep Learning: Case studies and discussion.

Advanced case studies from areas of Science and Technology: Particle Physics, Astronomy and Astrophysics, Medical Imaging, Accelerator Physics.

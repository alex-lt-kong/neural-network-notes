#  Regularizing Neural Networks 

* Deep learning neural networks are likely to quickly overfit a training dataset with few examples.

## Dropout

* Ensembles of neural networks (i.e., a large number of neural networks) with different model configurations are known to reduce overfitting, but require the additional computational expense of training and maintaining multiple models.
* A single model can be used to simulate having a large number of different network architectures by randomly dropping out nodes during **training**.
* Dropout is implemented per-layer in a neural network. It can be used with most types of layers, such as dense fully connected layers, convolutional layers, and recurrent layers such as the long short-term memory network layer.
* Dropout can be used after convolutional layers (e.g. Conv2D) and after pooling layers (e.g. MaxPooling2D).

## Weight Constraint

* Weight constraints provide an approach to reduce the overfitting of a deep learning neural network model on the training data and improve the performance of the model on new data. There are multiple types of weight constraints, such as maximum and unit vector norms.
* A simpler weight constraint is maximum norm (max_norm), it forces weights to have a magnitude at or below a given limit.

## Reference:
* [A Gentle Introduction to Dropout for Regularizing Deep Neural Networks](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
* [How to Reduce Overfitting With Dropout Regularization in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/)
* [How to Reduce Overfitting Using Weight Constraints in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/)
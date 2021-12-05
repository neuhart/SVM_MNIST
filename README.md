# SVM_MNIST

Statistical Classifiers like the support vector classifier or the k-nearest neighbour classifier have proven to be effective tools for classifying real-world data.
In this assignment I would like to develop the main theory behind the Maximal
Margin Classifier, the simplest support vector algorithm, and also introduce
the k-nearest neighbour algorithm. In the second part I am focusing on the
application of these classifiers to a real world problem, handwritten digit recognition. The basis of this experiment was the MNIST data set of handwritten
digits, which is publicly available. I used small subsets of 1,5 and 10 examples
per digit and created new training examples through shifts and rotations of
these images to obtain a larger training set, and thus, hopefully improved test
accuracy. Furthermore, I tried to find small training sets, called super-trainers,
which give rise to particularly good test performances. Finally, I combined my
preprocessing approaches and was able to score test accuracy rates significantly
higher than the mean score of randomly assembled training sets.
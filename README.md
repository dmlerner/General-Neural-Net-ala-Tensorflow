# General-Neural-Net-ala-Tensorflow
  Compile with "g++ driver.cpp -o neural.out -O3 -std=c++11 -fopenmp -Wno-narrowing"
  
  This is a neuralnet library enabling simple creation and backpropagation training. It is specifically designed for image recognition, but could be repurposed broadly.
  
  Project is still in progress. Current version runs to typically 95% training and 91% validation accuracy recognizing MNIST handwritten digit dataset. Estimated completion February '17.
  
  A very general range of neural nets can be represented as graphs, like in the notable [Tensorflow library] (https://www.tensorflow.org/tutorials/mnist/tf/). Nodes store intermediate calculations, while edges feed these values into their children as inputs. For example, consider a neural net with weights w1 and w2, nonlinear activation function sigma, and input x: neuralNet(x) = w2 * sigma(w1 * x). Each of x, sigma, w1, and w2 are nodes. X and w1 are parents of (inputs to) sigma; sigma is a child (output) of x and w1. Similarly, w2 is a child of sigma. x and neuralNet can be imagined to be scalar or vector; correspondingly, w1 and w2 would be scalars or matrices. 
  
  One specifies a neural net archiecture in driver.cpp using the various operation structs from neural.cpp. All operation classes are or inherit from Node, which has scalar value. Structs named r1, r2, and r3 are rank 1, 2, and 3 tensors for storing weights. Specific operations inherit from those structs and add the ability to compute functions of any number and rank of inputs. 
  
  See also simpleDriver.cpp and simple.out for a simple architecture example. It implements a one operation, linear neural net, but is surprisingly effective! 
  


  


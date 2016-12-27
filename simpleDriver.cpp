#include "neural.cpp"

int main(int argc, char *argv[]) {
  readFiles();

  r2 x(28,28,"x"); //input image, initialized in line
  r3 w(10,28,28,"w"); //input weights, randomly initialized
  MatrixProduct2_1 product(w, x, "Matrix multiply"); //maps a 28x28 matrix to a length 10 vector
  Softmax s(product,"s");
  r1 t(10,"t");
  Entropy entropy(s, t, "entropy");
  NeuralNet brain(x);
  brain.topologicSort();
  brain.printNames();

  int epochs=1000000;
  int printSkip=50;
  double learningRate = 1e-2;

  verbose = false;

  i1 order = getRandomInts(epochs, 1, xTrain.size()-1);
  cout << "starting training..." << endl;
  for (int i=0; i<epochs; i++) {
    if (i==500)
      learningRate /= 5;
    if (i==2000)
      learningRate /= 5;
    x.state.d2 = xTrain[order[i]];
    t.state.d1 = tTrain[order[i]];
    brain.learn(learningRate);
    if (i%printSkip==0) {
      cout << i << " " << brain.percentCorrect(xTrain, &x, tTrain, &t, &s) << " " <<
           brain.percentCorrect(xTest, &x, tTest, &t, &s);
      cout << " " << brain.averageLoss(xTrain, &x, tTrain, &t, &entropy) << " " <<
           brain.averageLoss(xTest, &x, tTest, &t, &entropy) << " " << learningRate << " " << endl;
    }
  }
}

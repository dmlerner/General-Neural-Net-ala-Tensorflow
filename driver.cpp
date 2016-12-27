#include "neural.cpp"

int main(int argc, char *argv[]) {
  readFiles();
  p(xTrain.size());
  p(xTest.size());

  r2 x(28,28,"x");
  int convSize = 2;
  int nConvolutions = 5;
  //r2 convBias(14,14,"convBias");
  vector<Convolution*> convolutions;
  vector<r2*> convWeights, fullWeights;
  vector<MaxPool*> maxPools;
  vector<RELU*> relus;
  vector<Matrix2_2*> fullConnections;

  for (int i=0; i<nConvolutions; i++) {
    string s = to_string(i);
    convWeights.push_back(new r2(convSize, convSize, "wConv"+s,true));
    convolutions.push_back(new Convolution(*convWeights[i], x, "conv"+s));
    maxPools.push_back(new MaxPool(*convolutions[i], "max"+s));
    fullWeights.push_back(new r2(14,14,"fullWeight"+s, true));
    fullConnections.push_back(new Matrix2_2(*fullWeights[i], *maxPools[i], "full after conv"+s));
    relus.push_back(new RELU(*fullConnections[i], "relu"+s));
  }
  r1 wReluCombine(nConvolutions, "wReluCombine", true);
  Combine2 fullCombiner(&wReluCombine, relus, "full combiner");
  r3 M21(10,14,14,"m21 weights",true);
  MatrixProduct2_1 full(M21, fullCombiner, "M21 op");
  r1 bFull(10,"bFull",true);
  Add1 aFull(full, bFull, "aFull");
  Softmax s(aFull,"s");
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

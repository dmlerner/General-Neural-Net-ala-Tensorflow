#include <bits/stdc++.h>
using namespace std;
#define p(x) cout << #x << " = "<< x<< endl
#define l() cout << __FILE__ << " : " << __LINE__ << endl;

//#define min(a,b) a<b ? a : b
typedef vector<double> D1;
typedef vector<D1> D2;
typedef vector<D2> D3;
typedef vector<D3> D4;
typedef vector<D4> D5;
typedef vector<D5> D6;
typedef vector<int> i1;
int seed;
bool time_seed = true;
bool verbose = false;

D3 xTrain, xTest;
D2 tTrain, tTest;




int argmax(D1 x) {
  int maxIndex=0;
  double maxValue=x[0];
  for (int i=1; i<x.size(); i++) {
    if (x[i] > maxValue) {
      maxValue = x[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

void print(vector<int> x) {
  for (int i: x)
    cout << i << endl;
  cout << endl;
}

void print(D1 x) {
  for (double d: x)
    cout << d << endl;
  cout << endl;
}

void print(D2 x) {
  for (auto row: x) {
    for (double d: row) {
      cout << d << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void print(D3 x) {
  for (D2 X: x)
    print(X);
}

void print(D4 x) {
  for (D3 X: x)
    print(X);
}

void toRank2(D1&x, int rows, D2& y) {
  for (int i=0; i<x.size()/rows; i++) {
    y.emplace_back();
    for (int row=0; row<rows; row++) {
      y[i].push_back(x[i*rows+row]);
    }
  }
}

void toRank3(D1& x, int rows, int cols, D3& y) {
  for (int i=0; i<x.size()/rows/cols; i++) {
    y.emplace_back();
    for (int row=0; row<rows; row++) {
      y[i].emplace_back();
      for (int col=0; col<cols; col++) {
        y[i][row].push_back(x[i*rows*cols+row*cols+col]);
      }
    }
  }
}


void readFiles(int train=2000, int test=2000) {
  D1 _xTrain, _xTest;
  D1 _tTrain, _tTest;

  string line;
  int i=0;

  ifstream fXTrain("xTrain.csv");
  while(i++<train*pow(28,2) && getline(fXTrain, line) ) {
    //cout << i << endl;
    _xTrain.push_back(stod(line));
  }
  i=0;
  toRank3(_xTrain, 28, 28, xTrain);

  ifstream fXTest("xTest.csv");
  while(i++<test*pow(28,2) && getline(fXTest, line)) {
    _xTest.push_back(stod(line));
  }
  i=0;
  toRank3(_xTest, 28, 28, xTest);

  ifstream fTTrain("tTrain.csv");
  while(i++<train*10 && getline(fTTrain, line)) {
    _tTrain.push_back(stoi(line));
  }
  i=0;
  toRank2(_tTrain, 10, tTrain);

  ifstream fTTest("tTest.csv");
  while(i++<test*10 && getline(fTTest, line)) {
    _tTest.push_back(stoi(line));
  }
  i=0;
  toRank2(_tTest, 10, tTest);
}

D1 getRandomDoubles(int size, double mean=1, double standard_deviation=1) {
  static normal_distribution<double> distribution(mean, standard_deviation);
  if (time_seed)
    seed=time(NULL);
  //int seed=123; //123 works, 124 fails
  static default_random_engine generator(seed);
  D1 data(size);
  generate(data.begin(), data.end(), []() {
    return distribution(generator);
  });
  //  generate(data.begin(), data.end(), [](){return -.1;});
  return data;
}

D2 getRandomDoubles(int rows, int cols, double mean=0, double standard_deviation=1) {
  D1 d = getRandomDoubles(rows*cols, mean, standard_deviation);
  D2 e;
  toRank2(d, cols, e);
  return e;
}

D3 getRandomDoubles(int depth, int rows, int cols, double mean=0, double standard_deviation=1) {
  D1 d = getRandomDoubles(depth*rows*cols, mean, standard_deviation);;
  D3 e;
  toRank3(d, rows, cols, e);
  return e;
}

i1 getRandomInts(int n, int low, int high) {
  static uniform_int_distribution<int> distribution(low, high);
  if (time_seed)
    seed=time(NULL);
  static default_random_engine generator(seed);
  i1 random(n);
  generate(random.begin(), random.end(), []() {
    return distribution(generator);
  });
  return random;
}

D2 getRandomDoublesUniform(int r, int c, double low=0, double high=1) {
  static uniform_real_distribution<double> distribution(low, high);
  if (time_seed)
    seed=time(NULL);
  static default_random_engine generator(seed);
  D1 random(r*c);
  generate(random.begin(), random.end(), []() {
    return distribution(generator);
  });
  D2 randomGrid;
  toRank2(random, r, randomGrid);
  return randomGrid;
}



struct Tensor {
  double d0=0;
  D1 d1;
  D2 d2;
  D3 d3;
  D4 d4;
  D5 d5;
  D6 d6;

  int rank=-1;
  vector<int> shape;

  Tensor(bool _randomize=false) {
    //cout << "wrong constructor" << endl;
    rank = 0;
    if (_randomize) {
      randomize();
      scale(.1);
      cout << "scaled!" << endl;
    }
    else
      zero();
  }

  Tensor(vector<int> s, bool _randomize=false) {
    shape = s;
    rank = s.size();
    if (_randomize) {
      randomize();
      scale(.1);
    }
    else
      zero();
  }

  Tensor(int x, bool _randomize=false) {
    shape = vector<int> {x};
    rank = 1;
    if (_randomize)
      randomize();
    else
      zero();
  }

  void randomize() {
    if (rank==0)
      d0 = getRandomDoubles(1)[0];
    else if (rank==1)
      d1 = getRandomDoubles(shape[0]);
    else if (rank==2)
      d2 = getRandomDoubles(shape[0], shape[1]);
    else if (rank==3)
      d3 = getRandomDoubles(shape[0], shape[1], shape[2]);
    else
      cout << "Tensor.randomize: invalid rank" << endl;
  }

  void zero() {
    if (rank==0)
      d0 = 0;
    else if (rank==1)
      d1 = D1(shape[0]);
    else if (rank==2)
      d2 = D2(shape[0], D1(shape[1]));
    else if (rank==3)
      d3 = D3(shape[0], D2(shape[1], D1(shape[2])));
    else if (rank==4)
      d4 = D4(shape[0], D3(shape[1], D2(shape[2], D1(shape[3]))));
    else if (rank==5)
      d5 = D5(shape[0], D4(shape[1], D3(shape[2], D2(shape[3], D1(shape[4])))));
    else if (rank==6)
      d6 = D6(shape[0], D5(shape[1], D4(shape[2], D3(shape[3], D2(shape[4], D1(shape[5]))))));
    else
      assert (false);
  }

  void addMultiple(Tensor& t, double multiple=1) {
    assert (shape == t.shape);
    if (rank==0)
      d0 += multiple*t.d0;

    else if (rank==1)
      for (int i=0; i<shape[0]; i++)
        d1[i] += multiple*t.d1[i];

    else if (rank==2)
      for (int i=0; i<shape[0]; i++)
        for (int j=0; j<shape[1]; j++)
          d2[i][j] += multiple*t.d2[i][j];

    else if (rank==3)
      for (int i=0; i<shape[0]; i++)
        for (int j=0; j<shape[1]; j++)
          for (int k=0; k<shape[2]; k++)
            d3[i][j][k] += multiple*t.d3[i][j][k];

    else {
      cout << "Tensor.addMultiple not implemented for rank >3" << endl;
      assert (false);
    }
  }

  void scale(double factor) {
    addMultiple(*this, factor-1);
  }

  double fix(double& x, double cutoff=100) {
    if (isnan(x))
    {
      assert (false);

      x=0;
    }
    if (x > cutoff)
      x= cutoff;
    if (x < -cutoff)
      x= -cutoff;
  }

  void clip(double cutoff=1) {
    if (rank==0)
      fix(d0);

    else if (rank==1)
      for (int i=0; i<shape[0]; i++)
        fix(d1[i]);

    else if (rank==2)
      for (int i=0; i<shape[0]; i++)
        for (int j=0; j<shape[1]; j++)
          fix(d2[i][j]);

    else if (rank==3)
      for (int i=0; i<shape[0]; i++)
        for (int j=0; j<shape[1]; j++)
          for (int k=0; k<shape[2]; k++)
            fix(d3[i][j][k]);
  }

  void printState() {
    if (rank==0)
      cout << d0 << endl << endl;
    else if (rank==1)
      print(d1);
    else if (rank==2)
      print(d2);
    else if (rank==3)
      print(d3);
    else if (rank==4)
      print(d4);
    else {
      cout << "invalid rank in printState" << endl;
      cout << rank << endl;
      assert (false);
    }
  }
};

struct Node {
  vector<Node*> parents, children;
  Tensor state;
  vector<Tensor> gradients;
  bool marked=false, tempMarked=false;
  string name;
  Tensor lossGradient;
  int topoDepth=-1;

  void addChild(Node& n) {
    children.push_back(&n);
    n.parents.push_back(this);
  }

  void addParent(Node& n) {
    parents.push_back(&n);
    n.children.push_back(this);
  }

  void addChild(Node* n) {
    children.push_back(n);
    n->parents.push_back(this);
  }

  void addParent(Node* n) {
    parents.push_back(n);
    n->children.push_back(this);
  }

  Node(string _name="") {
    name = _name;
    lossGradient.rank = 0;
    lossGradient.d0 = 1;
  }

//  Node(Node& n, string _name="") {
//    n.addChild(this);
//    name = _name;
//    lossGradient.rank = 0;
//    lossGradient.d0 = 1;
//  }
  Node(Node* n, string _name="") {
    cout << "in suspicious node constructor" << endl;
    n->addChild(this);
    name = _name;
    lossGradient.rank = 0;
    lossGradient.d0 = 1;
  }
  void scale(double multiple) {
    state.scale(multiple);
  }

  virtual void updateState() {}

  virtual void computeGradient() {}

  virtual void backwardPropagate() {
    if (verbose) {
      cout << "backpropagating from: ";
      p(name);
    }

    for (int p=0; p<parents.size(); p++) {
      int r = parents[p]->state.rank;
      if (verbose) {
        p(parents[p]->name);
        p(parents[p]->state.rank);
        print(parents[p]->state.shape);
        cout << "the gradient: " << endl;
        gradients[p].printState();

        p(name);
        p(lossGradient.d0);
      }

      if (r==0)
        parents[p]->lossGradient.d0 += lossGradient.d0*gradients[p].d0;

      else if (r==1) {

        for (int j=0; j<parents[p]->state.shape[0]; j++)
          parents[p]->lossGradient.d1[j] += lossGradient.d0*gradients[p].d1[j];

        if (verbose) {
          cout << "parent loss gradient freshly computed" << endl;
          p(parents[p]->name);
          parents[p]->lossGradient.printState();
        }
      }

      else if (r==2)
        for (int j=0; j<parents[p]->state.shape[0]; j++)
          for (int k=0; k<parents[p]->state.shape[1]; k++)
            parents[p]->lossGradient.d2[j][k] += lossGradient.d0*gradients[p].d2[j][k];

      else if (r==3)
        for (int j=0; j<parents[p]->state.shape[0]; j++)
          for (int k=0; k<parents[p]->state.shape[1]; k++)
            for (int l=0; l<parents[p]->state.shape[2]; l++)
              parents[p]->lossGradient.d3[j][k][l] += lossGradient.d0*gradients[p].d3[j][k][l];

      else
        assert (false);
      //return;
      //
      if (verbose) {
        cout << "loss gradient just computed" << endl;
        p(parents[p]->name);
        parents[p]->lossGradient.printState();
      }
    }
  }


  void learn(double learningRate=.01, double cutoff=0) {
    if (verbose) {
      p(name);
      cout << "lossGrad at learning time: " << endl;
      lossGradient.printState();
    }
    //in each of node, r1, r2, r3, implement L1 clipping
    lossGradient.clip(cutoff);
    state.addMultiple(lossGradient, -learningRate);
  }
};


struct r1:Node {
  int r;

  r1() {}

  r1(int R, string _name="", bool randomize=false) {
    r=R;
    name = _name;
    state = Tensor(i1{R}, randomize);
    lossGradient = Tensor(i1{R});
  }

  void backwardPropagate() override {
    //surely there is a clever recursive way to do this without all ifs
    //and without need to write a slightly modified version in r2, r3...
    //    p(name);
    if (verbose) {
      cout << "backwardPropagate from" << endl;
      p(name);
      cout << "lossGradient" << endl;
      lossGradient.printState();
    }

    for (int p=0; p<parents.size(); p++) {
      int r = parents[p]->state.rank;

      for (int i=0; i<lossGradient.shape[0]; i++) {
        if (r==0)
          parents[p]->lossGradient.d0 += lossGradient.d1[i]*gradients[p].d1[i];

        else if (r==1)
          for (int j=0; j<parents[p]->state.shape[0]; j++)
            parents[p]->lossGradient.d1[j] += lossGradient.d1[i]*gradients[p].d2[i][j];

        else if (r==2) {
          if (verbose && i==0) {
            cout << "in r1.backprop to r2" << endl;
            p(name);
            p(parents[p]->name);
            cout << "r1 loss grad:" << endl;
            lossGradient.printState();
          }
          for (int j=0; j<parents[p]->state.shape[0]; j++)
            for (int k=0; k<parents[p]->state.shape[1]; k++)
              parents[p]->lossGradient.d2[j][k] += lossGradient.d1[i]*gradients[p].d3[i][j][k];
        }

        else if (r==3)
          for (int j=0; j<parents[p]->state.shape[0]; j++)
            for (int k=0; k<parents[p]->state.shape[1]; k++)
              for (int l=0; l<parents[p]->state.shape[2]; l++)
                parents[p]->lossGradient.d3[j][k][l] += lossGradient.d1[i]*gradients[p].d4[i][j][k][l];

        else
          assert (false);
        //return;
      }
    }
  }
};

struct r2:Node {
  int r,c;

  r2() {
    r=c=0;
  }
  r2(int R, int C, string _name="", bool randomize=false): Node(_name) {
    r=R;
    c=C;
    state = Tensor(i1{r,c}, randomize);
    lossGradient = Tensor(i1{r,c}, false);
  }

  void backwardPropagate() override {
    for (int p=0; p<parents.size(); p++) {
      int r = parents[p]->state.rank;
      //#pragma omp parallel for
      for (int i=0; i<lossGradient.shape[0]; i++)
        for (int j=0; j<lossGradient.shape[1]; j++)
          if (r==0)
            parents[p]->lossGradient.d0 += lossGradient.d2[i][j]*gradients[p].d2[i][j];

          else if (r==1) {
            for (int k=0; k<parents[p]->state.shape[0]; k++) {
              parents[p]->lossGradient.d1[k] += lossGradient.d2[i][j]*gradients[p].d3[i][j][k];
            }
          }

          else if (r==2) {
            for (int k=0; k<parents[p]->state.shape[0]; k++)
              for (int l=0; l<parents[p]->state.shape[1]; l++)
                parents[p]->lossGradient.d2[k][l] += lossGradient.d2[i][j]*gradients[p].d4[i][j][k][l];
          }



          else if (r==3)
            for (int k=0; k<parents[p]->state.shape[0]; k++)
              for (int l=0; l<parents[p]->state.shape[1]; l++)
                for (int m=0; m<parents[p]->state.shape[2]; m++)
                  parents[p]->lossGradient.d3[k][l][m] += lossGradient.d2[i][j]*gradients[p].d5[i][j][k][l][m];

          else
            assert (false);
    }
  }
};

struct r3:Node {
  int r, c, d;
  r3() {} //I feel like I need this, but am not sure why
  r3(int R, int C, int D, string _name="", bool randomize=false) {
    r=R;
    c=C;
    d=D;
    name = _name;
    state = Tensor(i1{R,C,D}, randomize);
    lossGradient = Tensor(i1{R,C,D}, false);
  }

  void backwardPropagate() override {
    assert ("doo"=="doo");
    for (int p=0; p<parents.size(); p++) {

      int r = parents[p]->state.rank;
      for (int i=0; i<lossGradient.shape[0]; i++)
        for (int j=0; j<lossGradient.shape[1]; j++)
          for (int k=0; k<lossGradient.shape[2]; k++) {
            if (r==0)
              parents[p]->lossGradient.d0 += lossGradient.d3[i][j][k]*gradients[p].d3[i][j][k];

            else if (r==1)
              for (int l=0; l<parents[p]->state.shape[0]; l++)
                parents[p]->lossGradient.d1[l] += lossGradient.d3[i][j][k]*gradients[p].d4[i][j][k][l];

            else if (r==2)
              for (int l=0; l<parents[p]->state.shape[0]; l++)
                for (int m=0; m<parents[p]->state.shape[1]; m++)
                  parents[p]->lossGradient.d2[l][m] += lossGradient.d3[i][j][k]*gradients[p].d5[i][j][k][l][m];

            else if (r==3)
              for (int l=0; l<parents[p]->state.shape[0]; l++)
                for (int m=0; m<parents[p]->state.shape[1]; m++)
                  for (int n=0; n<parents[p]->state.shape[2]; n++)
                    parents[p]->lossGradient.d3[l][m][n] += lossGradient.d3[i][j][k]*gradients[p].d6[i][j][k][l][m][n];

            else
              assert (false);
          }
    }
  }

};

struct MatrixProduct1_1: r1 {
  MatrixProduct1_1(r2& W, r1& x, string name=""):r1(x.state.shape[0], name) {
    W.addChild(this);
    x.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }

  MatrixProduct1_1(r2* W, r1* x, string name=""):r1(x->state.shape[0], name) {
    W->addChild(this);
    x->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  void updateState() override {
    D2& W = parents[0]->state.d2;
    D1& x = parents[1]->state.d1;
    state.zero();
    for (int i=0; i<W.size(); i++)
      for (int j=0; j<W[0].size(); j++)
        state.d1[i] += W[i][j]*x[j];
  }

  void computeGradient() override {
    D2& W = parents[0]->state.d2;
    D1& x = parents[1]->state.d1;
    gradients[0] = Tensor(vector<int> {r, W.size(), W[0].size()});
    gradients[1] = Tensor(vector<int> {r, x.size()});

    for (int i=0; i<r; i++)
      for (int j=0; j<W.size(); j++)
        for (int k=0; k<W[0].size(); k++)
          gradients[0].d3[i][j][k] = (i==j)*x[k];

    for (int i=0; i<r; i++)
      for (int j=0; j<x.size(); j++)
        gradients[1].d2[i][j] = W[i][j];
  }
};

struct MatrixProduct2_1: r1 {
  MatrixProduct2_1(r3& W, r2& x, string name=""):r1(W.state.shape[0], name) {
    W.addChild(this);
    x.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  MatrixProduct2_1(r3* W, r2* x, string name=""):r1(W->state.shape[0], name) {
    W->addChild(this);
    x->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }

  void updateState() override {
    D3& W = parents[0]->state.d3;
    D2& x = parents[1]->state.d2;
    state.zero();
    for (int i=0; i<W.size(); i++)
      for (int j=0; j<W[0].size(); j++)
        for (int k=0; k<W[0][0].size(); k++)
          state.d1[i] += W[i][j][k]*x[j][k];
  }

  void computeGradient() override {
    D3& W = parents[0]->state.d3;
    D2& x = parents[1]->state.d2;
    gradients[0] = Tensor(i1{r, W.size(), W[0].size(), W[0][0].size()});
    gradients[1] = Tensor(i1{r, x.size(), x[0].size()});
    for (int i=0; i<r; i++)
      for (int j=0; j<W.size(); j++)
        for (int k=0; k<W[0].size(); k++)
          for (int l=0; l<W[0][0].size(); l++)
            gradients[0].d4[i][j][k][l] = (i==j)*x[k][l]; //not sparse, but redundant
    for (int i=0; i<r; i++)
      for (int j=0; j<x.size(); j++)
        for (int k=0; k<x[0].size(); k++) {
          gradients[1].d3[i][j][k] = W[i][j][k];
        }
//				gradients[1].d3.at(i).at(j).at(k)=W.at(i).at(j).at(k);
  }
};

struct Convolution: r2 {
  Convolution(r2& W, r2& x, string name="", bool randomize=true): r2(x.state.shape[0], x.state.shape[1], name, randomize) {
    W.addChild(this);
    x.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  Convolution(r2* W, r2* x, string name="", bool randomize=true): r2(x->state.shape[0], x->state.shape[1], name, randomize) {
    W->addChild(this);
    x->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  void updateState() override {
    state.zero();

    D2& W = parents[0]->state.d2;
    D2& x = parents[1]->state.d2;
    int wCenterX = W[0].size() / 2;
    int wCenterY = W.size() / 2;
    int rows = x.size(), cols = x[0].size();
    int wRows = W.size(), wCols = W[0].size();

    int mm, nn, ii, jj;
    #pragma omp parallel for
    for(int i=0; i < rows; i++)
      for(int j=0; j < cols; j++)
        for(int m=0; m < W.size(); m++) {
          mm = W.size() - 1 - m;
          for(int n=0; n < wCols; n++) {
            nn = wCols - 1 - n;
            ii = i + (m - wCenterY);
            jj = j + (n - wCenterX);
            if (ii >= 0 && ii < rows && jj >= 0 && jj < cols)
              state.d2[i][j] += x[ii][jj] * W[mm][nn];
          }
        }
  }

  void computeGradient() override {
    D2& W = parents[0]->state.d2;
    D2& x = parents[1]->state.d2;
    int rW = W.size();
    int cW = W[0].size();
    int rX = x.size();
    int cX = x[0].size();
    gradients[0] = Tensor(i1{rX,cX,rW,cX});
    gradients[1] = Tensor(i1{rX,cX,rX,cX});
    //let's put output indices first.
    for (int i=0; i<rX; i++)
      for (int j=0; j<cX; j++)
        for (int k=0; k<rW; k++)
          for (int l=0; l<cW; l++) {
            if (i-rW/2+k>=0 && i-rW/2+k<rX)
              if (j-cW/2+l>=0 && j-cW/2+l<cX)
                gradients[0].d4[i][j][k][l] = x[i-rW/2+k][j-cW/2+l];
          }
    Tensor unflipped=gradients[0];
    for (int i=0; i<rX; i++)
      for (int j=0; j<cX; j++)
        for (int k=0; k<rW; k++)
          for (int l=0; l<cW; l++) {
            gradients[0].d4[i][j][k][l]=unflipped.d4[i][j][rW-1-k][cW-1-l];
          }

    for (int i=0; i<rX; i++)
      for (int j=0; j<cX; j++)
        for (int k=0; k<rW; k++)
          for (int l=0; l<cW; l++) {
            int kk=i+rW/2-k;
            int ll=j+cW/2-l;
            if (kk>=0 && kk<rW && ll>=0 && ll<cW)
              gradients[1].d4[i][j][k][l] = W[kk][ll];
          }
  }
};
struct RELU: r2 {
  double leakiness=0;
  RELU(r2& x, string name=""):r2(x.state.shape[0], x.state.shape[1], name) {
    x.addChild(this);
    gradients.emplace_back();
  }
  RELU(r2* x, string name=""):r2(x->state.shape[0], x->state.shape[1], name) {
    x->addChild(this);
    gradients.emplace_back();
  }
  void updateState() override {
    state.zero();
    D2& x = parents[0]->state.d2;
    for (int i=0; i<x.size(); i++)
      for (int j=0; j<x[0].size(); j++)
        if (x[i][j] > 0)
          state.d2[i][j] = x[i][j];
        else
          state.d2[i][j] = leakiness*x[i][j];
  }

  void computeGradient() override {
    gradients[0] = Tensor(i1{r, c, r, c});
    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++)
        if (state.d2[i][j]>0)
          gradients[0].d4[i][j][i][j] = 1;
        else
          gradients[0].d4[i][j][i][j] = leakiness;
  }


};

struct Softmax: r1 {
  Softmax(r1& x, string name=""):r1(x.state.shape[0], name) {
    x.addChild(this);
    gradients.emplace_back();
  }

  Softmax(r1* x, string name=""):r1(x->state.shape[0], name) {
    x->addChild(this);
    gradients.emplace_back();
  }

  void updateState() override {
    //state.zero();
    D1& x = parents[0]->state.d1;
    double largest = x[argmax(x)];
    double lndenom = largest;
    double expsum = 0;
    for (int i=0; i<x.size(); i++)
      expsum += exp(x[i]-largest);
    for (int i=0; i<x.size(); i++)
      state.d1[i] = exp(x[i]-largest) / expsum;
  }

  void computeGradient() override {
    gradients[0] = Tensor(i1 {r,r});
    for (int i=0; i<r; i++)
      for (int j=0; j<r; j++)
        gradients[0].d2[i][j] = state.d1[i]*((i==j) - state.d1[j]);
    //BUG FIXED: need paren around i==j b/c == is after -. oy.
  }
};

struct Add0: Node { //adds abritrary number of Nodes (r0)
  Add0(vector<Node*> addends, string name=""): Node(name) {
    //this should work with Node& instead of Node* surely...
    for (Node* n: addends)
      n->addChild(this);
  }

  Add0(Node& a, Node& b, string name=""): Node(name) {
    a.addChild(this);
    b.addChild(this);
  }
  Add0(Node* a, Node* b, string name=""): Node(name) {
    a->addChild(this);
    b->addChild(this);
  }


  void updateState() override {
    state.zero();
    for (Node* p: parents)
      state.d0 += p->state.d0;
  }

  void computeGradient() override {
    gradients = vector<Tensor>(parents.size());

    for (int i=0; i<parents.size(); i++) {
      gradients[i].rank = 0;
      gradients[i].d0 = 1;
    }
  }
};

struct Add1: r1 {
  Add1(Node& x, Node& y, string name=""):r1(x.state.shape[0], name) { //adds 2 r1s. maybe i should generalize to n.
    x.addChild(this);
    y.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }
  Add1(Node* x, Node* y, string name=""):r1(x->state.shape[0], name) { //adds 2 r1s. maybe i should generalize to n.
    x->addChild(this);
    y->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }
  void updateState() override {
    D1& x = parents[0]->state.d1;
    D1& y = parents[1]->state.d1;
    for (int i=0; i<r; i++)
      state.d1[i] = x[i]+y[i];
  }

  void computeGradient() override {
    gradients[0] = Tensor(i1{r,r});
    gradients[1] = Tensor(i1{r,r});
    for (int i=0; i<r; i++) {
      gradients[0].d2[i][i] = 1;
      gradients[1].d2[i][i] = 1;
    }
  }

};

struct Add2: r2 {
  template <typename T>
  Add2(vector<T*> nodes, string name=""): r2(nodes[0]->state.shape[0], nodes[0]->state.shape[1], name) {
    for (int i=0; i<nodes.size(); i++) {
      nodes[i]->addChild(this);
      gradients.emplace_back();
    }
  }
  Add2(r2& x, r2& y, string name=""): r2(x.state.shape[0], x.state.shape[1], name) {
    x.addChild(this);
    y.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }
  Add2(r2* x, r2* y, string name=""): r2(x->state.shape[0], x->state.shape[1], name) {
    x->addChild(this);
    y->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  void updateState() override {
    state.zero();
    for (Node* p: parents) {

      for (int i=0; i<r; i++)
        for (int j=0; j<c; j++)
          state.d2[i][j] += p->state.d2[i][j];
    }
  }

  void computeGradient() override {
    for (int p=0; p<parents.size(); p++)
      gradients[p] = Tensor(i1{r,c,r,c});
    for (int p=0; p<parents.size(); p++)
      for (int i=0; i<r; i++)
        for (int j=0; j<c; j++)
          gradients[p].d4[i][j][i][j] = 1;
  }


};

struct Combine2: r2 {
  template <typename T>
  Combine2(r1* W, vector<T*> nodes, string name=""): r2(nodes[0]->state.shape[0], nodes[0]->state.shape[1], name) {
    gradients.emplace_back();
    W->addChild(this);
    for (int i=0; i<nodes.size(); i++) {
      nodes[i]->addChild(this);
      gradients.emplace_back();
    }
  }
  void updateState() override {
    state.zero();
    for (int p=0; p<parents.size()-1; p++)
      for (int i=0; i<r; i++)
        for (int j=0; j<c; j++) {
          state.d2[i][j] += parents[0]->state.d1[p]*parents[p+1]->state.d2[i][j];
        }
  }

  void computeGradient() override {
    for (int p=1; p<parents.size(); p++)
      gradients[p] = Tensor(i1{r,c,r,c});
    gradients[0] = Tensor(i1{state.shape[0], state.shape[1],parents.size()-1 });

    for (int p=1; p<parents.size(); p++)
      for (int i=0; i<r; i++)
        for (int j=0; j<c; j++)
          gradients[p].d4[i][j][i][j] = 1;

    for (int p=1; p<parents.size(); p++)
      for (int i=0; i<parents[p]->state.shape[0]; i++)
        for (int j=0; j<parents[p]->state.shape[1]; j++)
          gradients[0].d3[i][j][p-1] = parents[p]->state.d2[i][j];
  }
};

struct MaxPool: r2 {
  MaxPool(r2& x, string name=""): r2(x.state.shape[0]/2, x.state.shape[1]/2, name) {
    x.addChild(this);
    gradients.emplace_back();
  }
  MaxPool(r2* x, string name=""): r2(x->state.shape[0]/2, x->state.shape[1]/2, name) {
    x->addChild(this);
    gradients.emplace_back();
  }


  void updateState() override {
    D2& x = parents[0]->state.d2;
    //print(x);

    for (int i=0; i<x.size(); i+=2)
      for (int j=0; j<x[0].size(); j+=2) {
        state.d2[i/2][j/2] = max(max(x[i][j], x[i+1][j]), max(x[i+1][j], x[i+1][j+1]));
        // cout << state.d2[0][0];
        // exit(1);
      }
  }

  void computeGradient() override {
    //state.zero(); //WHY WAS THIS HERE, UGH BUGS BATMAN.
    D2& x = parents[0]->state.d2;
    gradients[0] = Tensor(i1{r,c,2*r,2*c});
    assert (r==gradients[0].d4.size());
    assert (2*r==gradients[0].d4[0][0].size());

    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++) {
        for (int k=i*2; k<min(i*2+2, 2*r); k++)
          for (int l=j*2; l<min(j*2+2, 2*c); l++)
            gradients[0].d4[i][j][k][l] = state.d2[i][j] == x[k][l]; //output indices first is my convention.
      }
  }
};
//
struct Dropout: r2 {
  double transmitProb;
  D2 rand;
  Dropout(r2& x, string name="", double p=.7): r2(x.state.shape[0], x.state.shape[1], name) {
    x.addChild(this);
    transmitProb=p;
    gradients.emplace_back();
  }
  Dropout(r2* x, string name="", double p=.7): r2(x->state.shape[0], x->state.shape[1], name) {
    x->addChild(this);
    transmitProb=p;
    gradients.emplace_back();
  }
  void updateState() override {
    state.zero();
    D2& x = parents[0]->state.d2;
    rand = getRandomDoublesUniform(r, c);
    for (int i=0; i<state.shape[0]; i++)
      for (int j=0; j<state.shape[1]; j++) {
        if (rand[i][j] > 1-transmitProb) {
          state.d2[i][j] = x[i][j];
        }

      }
  }

  void computeGradient() override {
    gradients[0] = Tensor(i1{r,c,r,c});
    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++)
        if (rand[i][j] > 1-transmitProb)
          gradients[0].d4[i][j][i][j] = 1;
  }

};

struct Entropy: Node {
  double epsilon = 1e-3; //don't take logs of zero!
  Entropy(r1& y, r1& t, string name=""): Node(name) {
    y.addChild(this);
    t.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
    lossGradient.d0 = 1;
  }
  Entropy(r1* y, r1* t, string name=""): Node(name) {
    y->addChild(this);
    t->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
    lossGradient.d0 = 1;
  }


  void updateState() override {
    state.zero();
    D1& y = parents[0]->state.d1;
    D1& t = parents[1]->state.d1;

    for (int i=0; i<y.size(); i++) {
      if (y[i]>1-epsilon) {
        //cout << "setting y to upper limit from " << y[i] << endl;
        y[i] = 1-epsilon;
      }
      if (y[i] < epsilon) {
        //cout << "setting y to lower limit from " << y[i] << endl;
        y[i] = epsilon;
      }
//      assert(y[i]!=0 && y[i]!=1);
//      if (!(y[i]>0)) {
//        cout << "loss: ";
//        p(y[i]);
//      }
//      //assert(y[i]>0);
      state.d0 -= t[i]*log(y[i]);
      state.d0 -= (1-t[i])*log(1-y[i]);
    }
  }

  void computeGradient() override {
    D1& y = parents[0]->state.d1;
    D1& t = parents[1]->state.d1;
    //cout << "about to initialize gradient" << endl;
    gradients[0] = Tensor(i1{y.size()});
    //cout << gradients[0].d1.size() << endl;
    //rxit(1);
    gradients[1] = Tensor(i1{t.size()}); //not fully general, which is probably ok
    // cout << y.size() << " " << t.size() << endl;
    // cout << gradients[0].d1.size() << endl;;
    // print(gradients[0].d1);
    // cout << "init done" << endl;
    //double derivativeClip = 10;
    for (int i=0; i<y.size(); i++) {
      if (y[i]>1-epsilon)
        y[i] = 1-epsilon;
      if (y[i] < epsilon)
        y[i] = epsilon;
      //assert(y[i]!=0 && y[i]!=1 && y[i]>0);
      gradients[0].d1[i] = -t[i]/y[i] + (1-t[i])/(1-y[i]);
      // if (abs(gradients[0].d1[i]) > derivativeClip)
      // gradients[0].d1[i] = derivativeClip * (gradients[0].d1[i] > 0 ? 1 : -1);
    }
  }
};

struct FrobeniusNorm: Node {

  FrobeniusNorm(Node& n, string name=""): Node(&n, name) {
    n.addChild(this);
    gradients.emplace_back();
    state.d0 = 0;
  };
  FrobeniusNorm(Node* n, string name=""): Node(n, name) {
    n->addChild(this);
    gradients.emplace_back();
    state.d0 = 0;
  };


  void updateState() override {
    state.zero();
    int r = parents[0]->state.rank;

    if (r==0)
      state.d0 = pow(parents[0]->state.d0,2);

    if (r==1)
      for (double d: parents[0]->state.d1)
        state.d0 += pow(d, 2);

    if (r==2)
      for (D1 d1: parents[0]->state.d2)
        for (double d: d1)
          state.d0 += pow(d, 2);

    if (r==3)
      for (D2 d2: parents[0]->state.d3)
        for (D1 d1: d2)
          for (double d: d1)
            state.d0 += pow(d, 2);
  }

  void computeGradient() {
    // p(parents.size());
    // p(parents[0]->name);
    gradients[0] = Tensor(parents[0]->state.shape);

    int r = parents[0]->state.rank;
    //    p(r);


    if (r==0)
      gradients[0].d0 += pow(parents[0]->state.d0,2);

    // if (r==1)
    // for (int i=0; i<gradients[0].shape[0]; i++)
    // gradients[0].d1[i] += pow(parents[0]->state.d1[i]*2, 2);
    //
    if (r==2)
      for (int i=0; i<gradients[0].shape[0]; i++)
        for (int j=0; j<gradients[0].shape[1]; j++)
          gradients[0].d2[i][j] += pow(parents[0]->state.d2[i][j]*2, 2);
    //cout << pow(parents[0]->state.d2[i][j]*2, 2);

    if (r==3)
      for (int i=0; i<gradients[0].shape[0]; i++)
        for (int j=0; j<gradients[0].shape[1]; j++)
          for (int k=0; k<gradients[0].shape[2]; k++)
            gradients[0].d3[i][j][k] += pow(parents[0]->state.d3[i][j][k]*2, 2);
    // ;
  }

};

struct Multiply: Node {
  double alpha;
  Multiply(Node& n, double _alpha=1, string name=""): Node(&n, name) {
    n.addChild(this);
    alpha = _alpha;
    gradients.emplace_back();
    computeGradient();
  }
  Multiply(Node* n, double _alpha=1, string name=""): Node(n, name) {
    n->addChild(this);
    alpha = _alpha;
    gradients.emplace_back();
    computeGradient();
  }


  void updateState() override {
    state.d0 = parents[0]->state.d0*alpha;
  }

  void computeGradient() override {
    gradients[0].d0 = alpha;
  }
};


void speak(Node& x) {
  cout << " in speak " << endl;
  p(x.name);
}
struct Dummy {
  Dummy(Node& x) {}
};

struct NeuralNet {
  vector<Node*> nodes;

  NeuralNet(Node& x) {
    nodes = getAll(x);
  }

  vector<Node*> getAll(Node& x) {
    vector<Node*> frontier;
    frontier.push_back(&x);
    p(frontier[0]->name);
    p(&x);
    p(frontier[0]);
    vector<Node*> all;
    while(frontier.size()) {
      for (Node* n: frontier) p(n);
      cout << endl << "starting while" << endl;
      p(frontier.size());
      //all.push_back(frontier.front());
      all.push_back(frontier[0]);
      p(frontier[0]);
      p(frontier[0]->state.rank);
      p(frontier[0]->name);
      for (int i=0; i<frontier[0]->children.size(); i++) {
        Node* child = frontier[0]->children[i];
        // for (Node* child: frontier.front()->children) {
        if (find(frontier.begin(), frontier.end(), child) == frontier.end())
          if (find(all.begin(), all.end(), child) == all.end()) {
            frontier.push_back(child);
          }
      }
      for (Node* parent: frontier.front()->parents) {
        if (find(frontier.begin(), frontier.end(), parent) == frontier.end())
          if (find(all.begin(), all.end(), parent) == all.end()) {
            frontier.push_back(parent);
          }
      }
      p(frontier[0]->name);
      p(frontier.size());
      frontier.erase(frontier.begin());
      p(frontier.size());
      p(frontier[1]->name);

    }
    return all;
  }


  void visit(Node* node, vector<Node*>& ordering) {
    assert (!node->tempMarked); //evidence you don't have a dag
    if (!node->marked) {
      node->tempMarked = true;
      for (Node* child: node->children)
        visit(child, ordering);
      node->marked = true;
      node->tempMarked = false;
      ordering.push_back(node);
    }
  }

  void topologicSort() { //somehow come sout backwards...
    vector<Node*> ordering;
    while (ordering.size() < nodes.size()) {
      Node* node;
      for (Node* n: nodes)
        if (!(n->tempMarked || n->marked)) {
          node = n;
          break;
        }
      visit(node, ordering);
    }
    nodes = ordering;
    reverse(nodes.begin(), nodes.end());
  }

  void forwardPropagate() {
    for (Node* n: nodes) {
      n->updateState();
      if (verbose) {
        cout << n->name << " forwardPropagate" << endl;
        n->state.printState();
      }

    }
  }

  void backwardPropagate() {
    reverse(nodes.begin(), nodes.end());

    for (Node* n: nodes)
      if (!(n->children.size()==0 && n->parents.size()!=0))// a final scalar output
        //the above line should maybe be controlled by the node itself...
        //but at leat its less awkward than checking the name
        n->lossGradient.zero();



    for (Node* n: nodes)
    {
      if(verbose) {
        cout << "backwards prop beginning from " << n->name << endl;
        p(n->name);
      }
      n->computeGradient();
      if(verbose)
        cout << "grad computed" << endl;
      if (n->parents.size())
        n->backwardPropagate();
      if (verbose) {
        cout << n->name  << " lossGrad" << endl;
        n->lossGradient.printState();
        if (n->name=="final loss") {
          cout << "final loss.parents[0].lossGradient" << endl;
          n->parents[0]->lossGradient.printState();
        }
        if (n->name=="b") {
          cout << "wrapping up b.backprop" << endl;
          n->lossGradient.printState();
        }
      }
      //cout << "backwardsPropagated" << endl;
    }
    reverse(nodes.begin(), nodes.end()); //now ordered with nodes[0] as input0
  }

  void printStates() {
    for (Node* n: nodes) {
      cout << n->name << endl;
      n->state.printState();
      cout << endl;
    }
  }

  void learn(double learningRate=.01) {
    forwardPropagate();
    backwardPropagate();
    for (Node* n: nodes) {
      if (n->parents.size() == 0 && n->name!="x" && n->name!="t") {
        // cout << "learning " << n->name << endl;
        // n->lossGradient.printState();
        n->learn(learningRate);
      }
    }
  }

  double averageLoss(D3& X, Node* x, D2& T, Node* t, Node* e) {
    double loss=0;
    for (int i=0; i<X.size(); i++) {
      x->state.d2 = X[i];
      t->state.d1 = T[i];
      forwardPropagate();
      loss += e->state.d0;
    }
    return loss/X.size();
  }

  double percentCorrect(D3& X, Node* x, D2& T, Node* t, Node* y) {
    double correct=0, incorrect=0;
    for (int i=0; i<X.size(); i++) {
      x->state.d2 = X[i];
      t->state.d1 = T[i];
      forwardPropagate();
      if (argmax(y->state.d1) == argmax(T[i])) {
        correct++;
      }
      else {
        incorrect++;
      }
    }
    return correct/(correct+incorrect);
  }

  // d1 percentCorrectHist(d3 X, d2 T){
  //   d1 correct(T[0].size());
  //   d1 incorrect(T[0].size());
  //   for (int i=0; i<X.size(); i++){
  //     Signal s = forwardPropagate(X[i]);
  //
  //     if (argmax(s.y) == argmax(T[i]))
  //     correct[argmax(T[i])] += 1.0;
  //     else
  //     incorrect[argmax(T[i])] += 1.0;
  //   }
  //   d1 percent(T[0].size());
  //   for (int i=0; i<T[0].size(); i++)
  //   if (correct[i] + incorrect[i])
  //   percent[i] = (double)correct[i]/(double)(correct[i]+incorrect[i]);
  //   return percent;
  // }

  void printNames() {
    for (Node* n: nodes)
      cout << n->name << endl;
  }
};

struct Matrix2_2: r2 {
  Matrix2_2(r2& a, r2& b, string name=""): r2(a.state.shape[0], a.state.shape[1], name) {
    a.addChild(this);
    b.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }
  Matrix2_2(r2* a, r2* b, string name=""): r2(a->state.shape[0], a->state.shape[1], name) {
    a->addChild(this);
    b->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
  }


  void updateState() override {
    D2 a=parents[0]->state.d2;
    D2 b=parents[1]->state.d2;
    state.zero();
    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++)
        for (int k=0; k<parents[0]->state.shape[1]; k++)
          state.d2[i][j] += a[i][k]*b[k][j];
  }

  void computeGradient() override {
    int K=parents[0]->state.shape[1];
    D2 a=parents[0]->state.d2;
    D2 b=parents[1]->state.d2;
    gradients[0] = Tensor(i1{r,c,r,K});
    gradients[1] = Tensor(i1{r,c,K,c}); //same shape
    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++)
        for (int m=0; m<K; m++)
          gradients[0].d4[i][j][i][m] = b[m][j];
    for (int i=0; i<r; i++)
      for (int j=0; j<c; j++)
        for (int l=0; l<K; l++)

          gradients[1].d4[i][j][l][j] = a[i][l];
  }

};
struct EuclideanDistance: Node {
  EuclideanDistance(r1& a, r1& b, string name=""): Node(name) {
    a.addChild(this);
    b.addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
    state.d0 = 0;
  }
  EuclideanDistance(r1* a, r1* b, string name=""): Node(name) {
    a->addChild(this);
    b->addChild(this);
    gradients.emplace_back();
    gradients.emplace_back();
    state.d0 = 0;
  }


  void updateState() override {
    state.zero();
    D1 a = parents[0]->state.d1;
    D1 b = parents[1]->state.d1;
    for (int i=0; i<a.size(); i++)
      state.d0 += pow(a[i]-b[i], 2);

  }

  void computeGradient() {

    gradients[0] = Tensor(parents[0]->state.shape);
    gradients[1] = Tensor(parents[1]->state.shape);
    for (int i=0; i<parents[0]->state.shape[0]; i++) {
      gradients[0].d1[i] = 2*(parents[0]->state.d1[i]-parents[1]->state.d1[i]);
      gradients[1].d1[i] = -gradients[0].d1[i];
    }
  }
};

Tensor numericDerivative(Node& n, NeuralNet& net, Node& loss) {
  Tensor derivative(n.state.shape);
  net.forwardPropagate();
  double error1 = loss.state.d0;
  double error2;
  double dw = 1e-4;

  if (n.state.rank == 0) {
    n.state.d0 += dw;
    net.forwardPropagate();
    n.state.d0 -= dw;
    error2 = loss.state.d0;
    derivative.d0 = (error2-error1)/dw;
  }

  if (n.state.rank == 1) {
    cout << "this should print " << n.name << endl;
    for (int i=0; i<n.state.shape[0]; i++) {
      n.state.d1[i] += dw;
      net.forwardPropagate();//problem: this writes over changes made to the weight for non constant things.
      n.state.d1[i] -= dw;
      error2 = loss.state.d0;
      derivative.d1[i] = (error2-error1)/dw;
      p(error2);
      p(error1);
    }
  }

  if (n.state.rank == 2) {
    for (int i=0; i<n.state.shape[0]; i++)
      for (int j=0; j<n.state.shape[1]; j++) {
        n.state.d2[i][j] += dw;
        net.forwardPropagate();
        n.state.d2[i][j] -= dw;
        error2 = loss.state.d0;
        derivative.d2[i][j] = (error2-error1)/dw;
      }
  }

  if (n.state.rank == 3) {
    for (int i=0; i<n.state.shape[0]; i++)
      for (int j=0; j<n.state.shape[1]; j++)
        for (int k=0; k<n.state.shape[2]; k++) {
          n.state.d3[i][j][k] += dw;
          net.forwardPropagate();
          n.state.d3[i][j][k] -= dw;
          error2 = loss.state.d0;
          derivative.d3[i][j][k] = (error2-error1)/dw;
        }

    return derivative;

  }
}

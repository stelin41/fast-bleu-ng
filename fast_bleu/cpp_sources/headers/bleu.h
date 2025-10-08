#ifndef BLEU_CPP_H
#define BLEU_CPP_H
#include <string>
#include <vector>
#include "fraction.h"
#include "counter.h"
#include "custmap.h"

using namespace std;

class BLEU_CPP
{
public:
  ~BLEU_CPP();
  BLEU_CPP();
  BLEU_CPP(vector<vector<string>>, vector<vector<float>>, int, int, bool, bool);
  vector<vector<double>> get_score(vector<vector<string>>);
  void append_reference(const vector<string>& reference);

private:
  vector<vector<string> *> references;
  vector<vector<vector<string> *>> references_ngrams;
  vector<vector<Counter *>> references_counts;
  vector<CustomMap *> reference_max_counts;
  vector<vector<float>> weights;
  int smoothing_function;
  bool auto_reweight;
  int max_n;
  vector<int> ref_lens;
  int number_of_refs;
  int n_cores;
  bool verbose;

  void get_max_counts(int);
  void get_max_counts_old(int);
};

#endif
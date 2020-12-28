//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: node.cpp 1595 2007-02-24 10:18:32Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <stdlib.h>
#include <cmath>
#include "node.h"
#include "common.h"

namespace CRFPP {

void Node::calcAlpha() {
  alpha = 0.0;
  for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it) {
    alpha = logsumexp(alpha,
                      (*it)->cost +(*it)->lnode->alpha,
                      (it == lpath.begin()));
  }
  alpha += cost;
}

void Node::calcBeta() {
  beta = 0.0;
  for (const_Path_iterator it = rpath.begin(); it != rpath.end(); ++it) {
    beta = logsumexp(beta,
                     (*it)->cost +(*it)->rnode->beta,
                     (it == rpath.begin()));
  }
  beta += cost;
}

void Node::calcExpectation(double *expected, double Z, size_t size) const {
  /*
   计算边缘概率p(y_i|x) = (alpha * beta) / Z
   实现方式exp(alpha + beta - cost - Z)
   */
  const double c = std::exp(alpha + beta - cost - Z);
  for (const int *f = fvector; *f != -1; ++f) {
    expected[*f + y] += c;  //将l*p(y_i|x)设置到expected数组中去
  }
  for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it) {
    //将p(y_i-1, y_i | x)的边缘概率设置到expected数组中
    (*it)->calcExpectation(expected, Z, size);
  }
}
}

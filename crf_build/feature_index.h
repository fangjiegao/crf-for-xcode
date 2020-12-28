//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: feature_index.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_FEATURE_INDEX_H_
#define CRFPP_FEATURE_INDEX_H_

#include <vector>
#include <map>
#include <iostream>
#include "common.h"
#include "scoped_ptr.h"
#include "feature_cache.h"
#include "path.h"
#include "node.h"
#include "freelist.h"
#include "mmap.h"
#include "darts.h"

namespace CRFPP {
class TaggerImpl;

class Allocator {
 public:
  explicit Allocator(size_t thread_num);
  Allocator();
  virtual ~Allocator();

  char *strdup(const char *str);    //复制一个新字符串
  Path *newPath(size_t thread_id);  //根据线程ID创建path,一个path对应一个标注序列
  Node *newNode(size_t thread_id);  //根据线程ID创建Node
  void clear();                     //清除内存
  void clear_freelist(size_t thread_id);  //清楚某个线程的内存
  FeatureCache *feature_cache() const;    //返回缓存的Feature
  size_t thread_num() const;

 private:
  void init();

  size_t                       thread_num_;
  scoped_ptr<FeatureCache>     feature_cache_;
  scoped_ptr<FreeList<char> >  char_freelist_;
  scoped_array< FreeList<Path> > path_freelist_;
  scoped_array< FreeList<Node> > node_freelist_;
};

class FeatureIndex {
 public:
  static const unsigned int version = MODEL_VERSION;

  size_t size() const  { return maxid_; }
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return y_.size(); }
  const char* y(size_t i) const { return y_[i].c_str(); }
  void   set_alpha(const double *alpha) { alpha_ = alpha; }
  const float *alpha_float() { return alpha_float_; }
  const double *alpha() const { return alpha_; }
  void set_cost_factor(double cost_factor) { cost_factor_ = cost_factor; }
  double cost_factor() const { return cost_factor_; }

  void calcCost(Node *node) const;
  void calcCost(Path *path) const;

  bool buildFeatures(TaggerImpl *tagger) const;
  void rebuildFeatures(TaggerImpl *tagger) const;

  const char* what() { return what_.str(); }

  explicit FeatureIndex(): maxid_(0), alpha_(0), alpha_float_(0),
                           cost_factor_(1.0), xsize_(0),
                           check_max_xsize_(false), max_xsize_(0) {}
  virtual ~FeatureIndex() {}

  const char *getTemplate() const;

 protected:
  virtual int getID(const char *str) const = 0;
  const char *getIndex(const char *&p,
                       size_t pos,
                       const TaggerImpl &tagger) const;
  bool applyRule(string_buffer *os,
                 const char *pattern,
                 size_t pos, const TaggerImpl &tagger) const;

  mutable unsigned int      maxid_;               //最大特征ID
  const double             *alpha_;               //特征权重,double类型,w
  const float              *alpha_float_;         //特征权重,float类型,w
  double                    cost_factor_;         //折扣因子
  unsigned int              xsize_;       //每一行训练数据的列数,比如“confidence N B”,xsize_=3
  bool check_max_xsize_;
  mutable unsigned int      max_xsize_;          //每一行训练数据的最大列数
  std::vector<std::string>  unigram_templs_;     //unigram模版数组
  std::vector<std::string>  bigram_templs_;      //bigram模版数组
  std::vector<std::string>  y_;                  //标签集合，所有可能的单个标签集合
  std::string               templs_;             //模版字符串,将unigram和bigram拼成大字符串
  whatlog                   what_;               //日志
};

class EncoderFeatureIndex: public FeatureIndex {
 public:
  bool open(const char *template_filename,
            const char *model_filename);
  bool save(const char *filename, bool emit_textmodelfile);
  bool convert(const char *text_filename,
               const char *binary_filename);
  void shrink(size_t freq, Allocator *allocator);

 private:
  int getID(const char *str) const;
  bool openTemplate(const char *filename);
  bool openTagSet(const char *filename);
  mutable std::map<std::string, std::pair<int, unsigned int> > dic_;
};

class DecoderFeatureIndex: public FeatureIndex {
 public:
  bool open(const char *model_filename);
  bool openFromArray(const char *buf, size_t size);

 private:
  Mmap <char> mmap_;
  Darts::DoubleArray da_;
  int getID(const char *str) const;
};
}
#endif

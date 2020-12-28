//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.cpp 1601 2007-03-31 09:47:18Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <fstream>
#include "param.h"
#include "encoder.h"
#include "timer.h"
#include "tagger.h"
#include "lbfgs.h"
#include "common.h"
#include "feature_index.h"
#include "scoped_ptr.h"
#include "thread.h"

namespace CRFPP {
namespace {

inline size_t getCpuCount() {
  size_t result = 1;
#if defined(_WIN32) && !defined(__CYGWIN__)
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  result = si.dwNumberOfProcessors;
#else
#ifdef HAVE_SYS_CONF_SC_NPROCESSORS_CONF
  const long n = sysconf(_SC_NPROCESSORS_CONF);
  if (n == -1) {
    return 1;
  }
  result = static_cast<size_t>(n);
#endif
#endif
  return result;
}

unsigned short getThreadSize(unsigned short size) {
  if (size == 0) {
    return static_cast<unsigned short>(getCpuCount());
  }
  return size;
}

bool toLower(std::string *s) {
  for (size_t i = 0; i < s->size(); ++i) {
    char c = (*s)[i];
    if ((c >= 'A') && (c <= 'Z')) {
      c += 'a' - 'A';
      (*s)[i] = c;
    }
  }
  return true;
}
}

class CRFEncoderThread: public thread {
 public:
  TaggerImpl **x;  //Taggers数组的指针,就是指针的指针
  unsigned short start_i;  //该线程的掌握的Tagger起始坐标
  unsigned short thread_num;  //支持的线程数
  int zeroone;  //一个序列只要有错误预测,加且只加一,有多少个序列存在预测错误
  int err;  //序列预测错误的个数
  size_t size;   //Tagger的个数
  double obj;  //目标值 -log p(y|x)
  std::vector<double> expected;  //梯度

  void run() {
    obj = 0.0;
    err = zeroone = 0;
    std::fill(expected.begin(), expected.end(), 0.0);
    for (size_t i = start_i; i < size; i += thread_num) {
      obj += x[i]->gradient(&expected[0]);  //求解梯度,返回-log p(y|x), 得到要优化的值
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) {
        ++zeroone;
      }
    }
  }
};

bool runMIRA(const std::vector<TaggerImpl* > &x,
             EncoderFeatureIndex *feature_index,
             double *alpha,
             size_t maxitr,
             float C,
             double eta,
             unsigned short shrinking_size,
             unsigned short thread_num) {
  std::vector<unsigned char> shrink(x.size());
  std::vector<float> upper_bound(x.size());
  std::vector<double> expected(feature_index->size());

  std::fill(upper_bound.begin(), upper_bound.end(), 0.0);
  std::fill(shrink.begin(), shrink.end(), 0);

  int converge = 0;
  int all = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    all += x[i]->size();
  }

  for (size_t itr = 0; itr < maxitr; ++itr) {
    int zeroone = 0;
    int err = 0;
    int active_set = 0;
    int upper_active_set = 0;
    double max_kkt_violation = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
      if (shrink[i] >= shrinking_size) {
        continue;
      }

      ++active_set;
      std::fill(expected.begin(), expected.end(), 0.0);
      double cost_diff = x[i]->collins(&expected[0]);
      int error_num = x[i]->eval();
      err += error_num;
      if (error_num) {
        ++zeroone;
      }

      if (error_num == 0) {
        ++shrink[i];
      } else {
        shrink[i] = 0;
        double s = 0.0;
        for (size_t k = 0; k < expected.size(); ++k) {
          s += expected[k] * expected[k];
        }

        double mu = std::max(0.0, (error_num - cost_diff) / s);

        if (upper_bound[i] + mu > C) {
          mu = C - upper_bound[i];
          ++upper_active_set;
        } else {
          max_kkt_violation = std::max(error_num - cost_diff,
                                   max_kkt_violation);
        }

        if (mu > 1e-10) {
          upper_bound[i] += mu;
          upper_bound[i] = std::min(C, upper_bound[i]);
          for (size_t k = 0; k < expected.size(); ++k) {
            alpha[k] += mu * expected[k];
          }
        }
      }
    }

    double obj = 0.0;
    for (size_t i = 0; i < feature_index->size(); ++i) {
      obj += alpha[i] * alpha[i];
    }

    std::cout << "iter="  << itr
              << " terr=" << 1.0 * err / all
              << " serr=" << 1.0 * zeroone / x.size()
              << " act=" <<  active_set
              << " uact=" << upper_active_set
              << " obj=" << obj
              << " kkt=" << max_kkt_violation << std::endl;

    if (max_kkt_violation <= 0.0) {
      std::fill(shrink.begin(), shrink.end(), 0);
      converge++;
    } else {
      converge = 0;
    }

    if (itr > maxitr || converge == 2) {
      break;  // 2 is ad-hoc
    }
  }

  return true;
}

bool runCRF(const std::vector<TaggerImpl* > &x,
            EncoderFeatureIndex *feature_index,
            double *alpha,
            size_t maxitr,
            float C,
            double eta,
            unsigned short shrinking_size,
            unsigned short thread_num,
            bool orthant) {
  double old_obj = 1e+37;
  int    converge = 0;
  LBFGS lbfgs;
  std::vector<CRFEncoderThread> thread(thread_num);

  for (size_t i = 0; i < thread_num; i++) {
    thread[i].start_i = i;
    thread[i].size = x.size();
    thread[i].thread_num = thread_num;
    thread[i].x = const_cast<TaggerImpl **>(&x[0]);
    thread[i].expected.resize(feature_index->size());
  }

  size_t all = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    all += x[i]->size();
  }

  for (size_t itr = 0; itr < maxitr; ++itr) {  //迭代优化,maxitr次
    for (size_t i = 0; i < thread_num; ++i) {
      thread[i].start();
    }

    for (size_t i = 0; i < thread_num; ++i) {
      thread[i].join();
    }

    for (size_t i = 1; i < thread_num; ++i) {
      thread[0].obj += thread[i].obj;
      thread[0].err += thread[i].err;
      thread[0].zeroone += thread[i].zeroone;
    }

    for (size_t i = 1; i < thread_num; ++i) {
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].expected[k] += thread[i].expected[k];
      }
    }

    size_t num_nonzero = 0;
    if (orthant) {   // L1
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += std::abs(alpha[k] / C);
        if (alpha[k] != 0.0) {
          ++num_nonzero;
        }
      }
    } else {
      num_nonzero = feature_index->size();
      for (size_t k = 0; k < feature_index->size(); ++k) {
        thread[0].obj += (alpha[k] * alpha[k] /(2.0 * C));
        thread[0].expected[k] += alpha[k] / C;
      }
    }

    double diff = (itr == 0 ? 1.0 :
                   std::abs(old_obj - thread[0].obj)/old_obj);
    std::cout << "iter="  << itr
              << " terr=" << 1.0 * thread[0].err / all
              << " serr=" << 1.0 * thread[0].zeroone / x.size()
              << " act=" << num_nonzero
              << " obj=" << thread[0].obj
              << " diff="  << diff << std::endl;
    old_obj = thread[0].obj;

    if (diff < eta) {
      converge++;
    } else {
      converge = 0;
    }

    if (itr > maxitr || converge == 3) {
      break;  // 3 is ad-hoc
    }

    if (lbfgs.optimize(feature_index->size(),
                       &alpha[0],
                       thread[0].obj,
                       &thread[0].expected[0], orthant, C) <= 0) {
      return false;
    }
  }

  return true;
}

bool Encoder::convert(const char* textfilename,
                      const char *binaryfilename) {
  EncoderFeatureIndex feature_index;
  CHECK_FALSE(feature_index.convert(textfilename, binaryfilename))
      << feature_index.what();

  return true;
}

/**
 * 训练
 * @param templfile 模板文件
 * @param trainfile 训练文件
 * @param modelfile 模型文件
 * @param textmodelfile 是否输出文本形式的模型文件
 * @param maxitr 最大迭代次数
 * @param freq 特征最低频次
 * @param eta 收敛阈值
 * @param C cost-factor
 * @param thread_num 线程数
 * @param shrinking_size
 * @param algorithm 训练算法
 * @return 0
 */
bool Encoder::learn(const char *templfile, //模版文件
                    const char *trainfile, //训练文件
                    const char *modelfile, //模型文件
                    bool textmodelfile,    //是否是文本模型文件
                    size_t maxitr,         //最大迭代次数
                    size_t freq,           //最小特征数,用于过滤掉过少的特征
                    double eta,            //如果两次迭代值小于eta，训练完毕
                    double C,              //正则化调节因子
                    unsigned short thread_num, //线程数
                    unsigned short shrinking_size, //MIRA算法参数
                    int algorithm) {       //算法类型 MIRA CRF_L2 CRF_L1
  std::cout << COPYRIGHT << std::endl;
  //参数校验开始
  CHECK_FALSE(eta > 0.0) << "eta must be > 0.0";
  CHECK_FALSE(C >= 0.0) << "C must be >= 0.0";
  CHECK_FALSE(shrinking_size >= 1) << "shrinking-size must be >= 1";
  CHECK_FALSE(thread_num > 0) << "thread must be > 0";
    std::cout << thread_num << ":" << shrinking_size << ":" << eta << ":" << C << std::endl;
#ifndef CRFPP_USE_THREAD
  CHECK_FALSE(thread_num == 1)
      << "This architecture doesn't support multi-thrading";
#endif

  if (algorithm == MIRA && thread_num > 1) {
    std::cerr <<  "MIRA doesn't support multi-thrading. use thread_num=1"
              << std::endl;
  }
  //参数校验结束
  EncoderFeatureIndex feature_index;  //训练数据和模版文件转成特征，并提供模版特征
  Allocator allocator(thread_num);
  std::vector<TaggerImpl* > x; //所有具体的算法运算工作都由Tagger完成,一个训练数据对应一个Tagger

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(5);

#define WHAT_ERROR(msg) do {                                    \
    for (std::vector<TaggerImpl *>::iterator it = x.begin();    \
         it != x.end(); ++it)                                   \
      delete *it;                                               \
    std::cerr << msg << std::endl;                              \
    return false; } while (0)

  CHECK_FALSE(feature_index.open(templfile, trainfile))
      << feature_index.what();

  {
    progress_timer pg;

    std::ifstream ifs(WPATH(trainfile));
    CHECK_FALSE(ifs) << "cannot open: " << trainfile;

    std::cout << "reading training data: " << std::flush;
    size_t line = 0;
    while (ifs) {
      TaggerImpl *_x = new TaggerImpl();
      _x->open(&feature_index, &allocator);    //将feature_index和allocator的引用告知Tagger
      if (!_x->read(&ifs) || !_x->shrink()) {  //TaggerImpl从feature_index中读数据
        WHAT_ERROR(_x->what());
      }

      if (!_x->empty()) {
        x.push_back(_x);
      } else {
        delete _x;
        continue;
      }

      _x->set_thread_id(line % thread_num);  //设置线程ID,平均的分配线程资源

      if (++line % 100 == 0) {
        std::cout << line << ".. " << std::flush;
      }
    }

    ifs.close();
    std::cout << "\nDone!";
  }
  std::cout << "freq:" << freq << std::endl;
  feature_index.shrink(freq, &allocator); //将模版中提取的特征计数少于freq的特征删除
  std::cout << "feature_index.size():" << feature_index.size() << std::endl;
  std::vector <double> alpha(feature_index.size());           // parameter,要训练的参数的分配,一个特征对应一个参数
  std::fill(alpha.begin(), alpha.end(), 0.0);  //初始化为0.0
  feature_index.set_alpha(&alpha[0]);

  std::cout << "Number of sentences: " << x.size() << std::endl;
  std::cout << "Number of features:  " << feature_index.size() << std::endl;
  std::cout << "Number of thread(s): " << thread_num << std::endl;
  std::cout << "Freq:                " << freq << std::endl;
  std::cout << "eta:                 " << eta << std::endl;
  std::cout << "C:                   " << C << std::endl;
  std::cout << "shrinking size:      " << shrinking_size << std::endl;
  // 准备工作全部就绪
  progress_timer pg;

  switch (algorithm) {
    case MIRA:
      if (!runMIRA(x, &feature_index, &alpha[0],
                   maxitr, C, eta, shrinking_size, thread_num)) {
        WHAT_ERROR("MIRA execute error");
      }
      break;
    case CRF_L2:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, false)) {
        WHAT_ERROR("CRF_L2 execute error");
      }
      break;
    case CRF_L1:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, true)) {
        WHAT_ERROR("CRF_L1 execute error");
      }
      break;
  }

  for (std::vector<TaggerImpl *>::iterator it = x.begin();
       it != x.end(); ++it) {
    delete *it;
  }

  if (!feature_index.save(modelfile, textmodelfile)) {
    WHAT_ERROR(feature_index.what());
  }

  std::cout << "\nDone!";

  return true;
}

namespace {
const CRFPP::Option long_options[] = {
  {"freq",     'f', "1",      "INT",
   "use features that occuer no less than INT(default 1)" },
  {"maxiter" , 'm', "100000", "INT",
   "set INT for max iterations in LBFGS routine(default 10k)" },
  {"cost",     'c', "1.0",    "FLOAT",
   "set FLOAT for cost parameter(default 1.0)" },
  {"eta",      'e', "0.0001", "FLOAT",
   "set FLOAT for termination criterion(default 0.0001)" },
  {"convert",  'C',  0,       0,
   "convert text model to binary model" },
  {"textmodel", 't', 0,       0,
   "build also text model file for debugging" },
  {"algorithm",  'a', "CRF",   "(CRF|MIRA)", "select training algorithm" },
  {"thread", 'p',   "0",       "INT",
   "number of threads (default auto-detect)" },
  {"shrinking-size", 'H', "20", "INT",
   "set INT for number of iterations variable needs to "
   " be optimal before considered for shrinking. (default 20)" },
  {"version",  'v', 0,        0,       "show the version and exit" },
  {"help",     'h', 0,        0,       "show this help and exit" },
  {0, 0, 0, 0, 0}
};

/**
 * 训练CRF模型
 * @param param 参数
 * @return 0
 */
int crfpp_learn(const Param &param) {
  if (!param.help_version()) {
    return 0;
  }

  const bool convert = param.get<bool>("convert");
  const std::vector<std::string> &rest = param.rest_args();
  if (param.get<bool>("help") ||
      (convert && rest.size() != 2) || (!convert && rest.size() != 3)) {
    std::cout << param.help();
    return 0;
  }
  size_t count = rest.size();
  for (int i = 0; i < count; i++)
  {
      printf("rest[%d]: %s", i, rest[i].c_str());
      printf("\n");
  }
  const size_t         freq           = param.get<int>("freq");
  const size_t         maxiter        = param.get<int>("maxiter");
  const double         C              = param.get<float>("cost");
  const double         eta            = param.get<float>("eta");
  const bool           textmodel      = param.get<bool>("textmodel");
  const unsigned short thread         =
      CRFPP::getThreadSize(param.get<unsigned short>("thread"));
  const unsigned short shrinking_size
      = param.get<unsigned short>("shrinking-size");
  std::string salgo = param.get<std::string>("algorithm");
  printf("algorithm %s \n", salgo.c_str());

  CRFPP::toLower(&salgo);
  printf("algorithm %s \n", salgo.c_str());

  int algorithm = CRFPP::Encoder::MIRA;
  if (salgo == "crf" || salgo == "crf-l2") {
    algorithm = CRFPP::Encoder::CRF_L2;
  } else if (salgo == "crf-l1") {
    algorithm = CRFPP::Encoder::CRF_L1;
  } else if (salgo == "mira") {
    algorithm = CRFPP::Encoder::MIRA;
  } else {
    std::cerr << "unknown alogrithm: " << salgo << std::endl;
    return -1;
  }
  printf("algorithm枚举类的值 %d \n", algorithm);

  CRFPP::Encoder encoder;
    if (convert) { //将文本模式转为二进制模式
    printf("convert is (true) : %d \n", convert);
    if (!encoder.convert(rest[0].c_str(), rest[1].c_str())) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  } else {
    printf("convert is (false) : %d \n", convert);
      //rest: template_file, train_file, model_file
    if (!encoder.learn(rest[0].c_str(),
                       rest[1].c_str(),
                       rest[2].c_str(),
                       textmodel,
                       maxiter, freq, eta, C, thread, shrinking_size,
                       algorithm)) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  }

  return 0;
}
}  // namespace
}  // CRFPP

int crfpp_learn2(const char *argv) {
  CRFPP::Param param;
  param.open(argv, CRFPP::long_options);
  return CRFPP::crfpp_learn(param);
}

/**
 * 命令行式训练
 * @param argc 命令个数
 * @param argv 命令数组
 * @return 0表示正常执行,其他表示错误
 */
int crfpp_learn(int argc, char **argv) {
  printf("encoder.cpp>crfpp_learn,🐷");
  printf("\n");
  CRFPP::Param param;
  param.open(argc, argv, CRFPP::long_options);
  return CRFPP::crfpp_learn(param);
}


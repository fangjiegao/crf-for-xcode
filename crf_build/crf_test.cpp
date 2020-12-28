//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: crf_test.cpp 1587 2007-02-12 09:00:36Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include "crfpp.h"
#include "winmain.h"
#include <string>
#include <iostream>

int main(int argc, char **argv) {
  size_t con_zn_len = std::string("中").size();
  std::string word = "小明今天穿了一件红色上衣";
  size_t iSize = word.size();
  for (size_t i = 0; i<iSize/con_zn_len; i++){
    std::string s = word.substr(i*con_zn_len,con_zn_len);
    std::cout << iSize/con_zn_len << i*con_zn_len << s << ";" << std::endl;
  }
  std::cout << word << std::endl;
  // word[2]='a';
  // std::cout << word << std::endl;
  // return crfpp_test(argc, argv);
  return my_crfpp_test(argc, argv);
  // std::string r = my_crfpp_test(argc, argv);
  // return 0;
}

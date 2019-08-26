####  
- 去掉augmentor.cc里的rnd_block_mask等aug  
  cd augmentation_threads  
  rm -rf build  
  mkdir build  
  cmake ..  
  cp libaugmentation.so ..  
  

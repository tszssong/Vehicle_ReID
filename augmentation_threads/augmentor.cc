
#include "simple_thread_pool.h"
#include "random"

#include "cv.h"
#include "highgui.h"

#include "opencv2/opencv.hpp"


struct Aug_Params
{
  string strfn;
  cv::Mat matOut;
  condition_variable *p_cv;
  mutex *p_countmt;
  int *p_FinishCount;
  int needNum;
  int stdsize[2];
  float *pfImgOut;
  float randv;
  int *pdwRect;//size=4
  int *pcrop;//size=2
};

int showimage()
{
  cout << "hello world\n";
  cv::Mat img = cv::imread("/Users/starimpact/work/2.jpg");
  if (img.cols==0)
  {
    cout << "can not read image.\n";
    return 0;
  }
  cv::imshow("hi", img);
  cv::waitKey(0);
  return 0;
}

void do_augment_onethread(void *p);
extern "C" int do_augment_threads(char *pfns[], int num, 
                                  int stdH, int stdW, float *pfImgOut)
{
  const int cdwMaxTNum = 48;
  static dg::ThreadPool *psPool = NULL;
  if (psPool == NULL)
  {
    printf("Max Thread Number:%d\n", cdwMaxTNum);
    psPool = new dg::ThreadPool(cdwMaxTNum);
  }

  srand(static_cast<unsigned>(time(0)));

  vector<string> vecfn;
  for (int i = 0; i < num; i++)
  {
    vecfn.push_back(pfns[i]);
  }
  condition_variable cv;
  mutex countmt;
  int dwFinishCount = 0;
  int fnum = vecfn.size();
  Aug_Params *pParams = new Aug_Params[fnum];

  for (int fi=0; fi < fnum; fi++) 
  {
    string strfn = vecfn[fi];
    pParams[fi].strfn = strfn;
    pParams[fi].p_cv = &cv;
    pParams[fi].p_countmt = &countmt;
    pParams[fi].p_FinishCount = &dwFinishCount;
    pParams[fi].needNum = fnum;
    pParams[fi].stdsize[0] = stdH;
    pParams[fi].stdsize[1] = stdW;
//    pParams[fi].randv = randv;
    pParams[fi].pfImgOut = pfImgOut + fi * stdH * stdW * 3;
    psPool->enqueue(do_augment_onethread, (void*)&pParams[fi]);
  }

  unique_lock<mutex> waitlc(countmt);
  cv.wait(waitlc, [&dwFinishCount, &fnum](){return dwFinishCount==fnum;});

//  for (int fi = 0; fi < fnum; fi++)
//  {
//    cv::Mat &img = pParams[fi].matOut;
//    memcpy(pfImgOut + fi * stdH * stdW * 3,  
//           img.data, sizeof(float) * stdH * stdW * 3);
////    cv::imshow("hi", img);
////    cv::waitKey(0);
//  }

  delete []pParams;
  return 0;
}



int rnd_crop(cv::Mat &matIn);
int rnd_rotate(cv::Mat &matIn, float randv);
int normalize_img(cv::Mat &matIn);
int rnd_mask(cv::Mat &matIn);
int rnd_block_mask(cv::Mat &matIn);
int part_crop(cv::Mat &matIn, int *pcrop);

void do_augment_onethread(void *p)
{
  Aug_Params *pParam = (Aug_Params*)p;
  string &strfn = pParam->strfn; 
  mutex *p_countmt = pParam->p_countmt;
  condition_variable *p_cv = pParam->p_cv;
  int *p_FinishCount = pParam->p_FinishCount;
  int needNum = pParam->needNum;
  int stdH = pParam->stdsize[0];
  int stdW = pParam->stdsize[1];
  float randv = pParam->randv;
  cv::Mat &matOut = pParam->matOut;
  float *pfImgOut = pParam->pfImgOut;

  cv::Mat img = cv::imread(strfn);
  if (img.cols==0)
  {
    printf("Can not read image %s\n", strfn.c_str());

    unique_lock<mutex> countlc(*p_countmt);
    if (*p_FinishCount < needNum)
    {
      (*p_FinishCount)++;
    }
    if ((*p_FinishCount) == needNum)
    {
      p_cv->notify_all();
    } 
    countlc.unlock();

    return;
  }
 
//  printf("%s\n", strfn.c_str()); 
  //mask rows
//  int rnd0 = rand();
//  if (rnd0 < (RAND_MAX / 4) * 3)
  {
//    rnd_mask(img);
//    rnd_block_mask(img);
  }
  
  //crop
//  cout << "******************"<< endl;
//  rnd_crop(img);
  //reisze
  cv::resize(img, img, cv::Size(stdW, stdH));
  //normalize
  normalize_img(img);
  //rotate
//  rnd_rotate(img, randv);
  //flip
//  int rnd = rand();
//  if (rnd < RAND_MAX / 2)
//  {
//    cv::flip(img, img, 1);
//  }
//  img.copyTo(matOut);
  float *pfImg = (float*)img.data;

  float *pfOutR = pfImgOut;
  float *pfOutG = pfImgOut + stdH * stdW;
  float *pfOutB = pfImgOut + stdH * stdW * 2;
  for (int ri = 0; ri < stdH; ri++)
  {
    for (int ci = 0; ci < stdW; ci++)
    {
       int dwOft = ri * stdW + ci;
       pfOutR[dwOft] = pfImg[dwOft * 3 + 0];
       pfOutG[dwOft] = pfImg[dwOft * 3 + 1];
       pfOutB[dwOft] = pfImg[dwOft * 3 + 2];
    }
  }

  unique_lock<mutex> countlc(*p_countmt);
  if (*p_FinishCount < needNum)
  {
    (*p_FinishCount)++;
  }
  if ((*p_FinishCount) == needNum)
  {
    p_cv->notify_all();
  } 
  countlc.unlock();

  return;
}

float randMToN(float M, float N)
{
  return M + (rand() / (RAND_MAX/(N-M))) ;  
}

int rnd_crop(cv::Mat &matIn)
{
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  int adwHWs[4] = {dwH, dwH, dwW, dwW};
  for (int dwI = 0; dwI < 4; dwI++)
  {
    float frndv = randMToN(0, 10) / 100;
//    cout << frndv << endl; 
    adwHWs[dwI] *= frndv;
  }
  cv::Rect roi(adwHWs[2], adwHWs[0], 
               dwW-adwHWs[3]-adwHWs[2], 
               dwH-adwHWs[1]-adwHWs[0]);
  cv::Mat matROI(matIn, roi);
  matROI.copyTo(matIn);
  return 0;
}

int part_crop(cv::Mat &matIn, int *pcrop)
{
  int cropY1 = pcrop[0];
  int cropY2 = pcrop[1];

  if (cropY1 == 0 && cropY2 == 0)
	  return 0;

  if (cropY2 <= cropY1)
	  return 0;



 //int dwH = matIn.rows;
  int dwW = matIn.cols;
  //printf("roi begin\n");
 // printf("image size:(%d,%d)\n",matIn.cols, matIn.rows);
  //cv::Rect roi(0, cropY1, dwW, cropY2);
  cv::Rect roi;
  roi.x = 0;
  roi.y = cropY1;
  roi.width = dwW;
  roi.height = cropY2 - cropY1;
//  printf("roi position: %d,%d,%d,%d\n", 0, cropY1, dwW, cropY2 - cropY1);
  cv::Mat matROI(matIn, roi);
  matROI.copyTo(matIn);
//  if (matIn.cols == 0)
//	  printf("accident size: %d, %d, %d, %d\n",roi.x, roi.y, roi.width, roi.height);
//	  printf("%d, %d\n",cropY1, cropY2); 
//  }
  return 0;

 }

int part_crop1(cv::Mat &matIn, int*pcrop){
	if (pcrop[2]<6 || pcrop[3]<6)
		return 0;

	cv::Rect roi;
	roi.x = pcrop[0];
	roi.y = pcrop[1];
	roi.width = pcrop[2];
	roi.height = pcrop[3];

	cv::Mat matROI(matIn, roi);
	matROI.copyTo(matIn);
	return 0;
}

int rnd_rotate(cv::Mat &matIn, float randv)
{
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  
 // float rndv = randMToN(0, 60) - 30;
  cv::Mat matRot = cv::getRotationMatrix2D(cv::Point(dwW/2, dwH/2), randv, 1.0);
  cv::warpAffine(matIn, matIn, matRot, cv::Size(dwW, dwH));

  return 0;
}


int normalize_img(cv::Mat &matIn)
{ 
  	
  matIn.convertTo(matIn, CV_32FC3, 1.0, 0);
  cv::Mat matmean, matstdv;
  cv::meanStdDev(matIn, matmean, matstdv);
//  cout << matmean.total() << "," << matstdv.total() << endl;
  float fmean = matmean.at<double>(0) + matmean.at<double>(1) + matmean.at<double>(2);
  fmean /= 3.0f;
  float fstdv = matstdv.at<double>(0) + matstdv.at<double>(1) + matstdv.at<double>(2);
  fstdv /= 3.0f;
//  cout << fmean << "," << fstdv << endl;
  matIn.convertTo(matIn, -1, 1.0f/fstdv, -fmean/fstdv);
//  cv::meanStdDev(matIn, matmean, matstdv);
//  cout << matmean << matstdv << endl;

  return 0;
}


int rnd_mask(cv::Mat &matIn)
{
  assert(matIn.type()==CV_8UC3);
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  
  int rndH = (int)randMToN(dwH/8, dwH/4);
  int rndRI = (int)randMToN(0, dwH-1);
  if (rndH + rndRI >= dwH)
  {
    rndRI = dwH - rndH - 1;
  }
  
  memset(matIn.data + rndRI * dwW * 3, 0, rndH * dwW * 3);

  return 0;
}


int rnd_block_mask(cv::Mat &matIn)
{
  assert(matIn.type()==CV_8UC3);
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  
  int rndH = (int)randMToN(dwH/4, dwH/2);
  int rndW = (int)randMToN(dwW/4, dwW/2);
  int rndRI = (int)randMToN(0, dwH-1);
  int rndCI = (int)randMToN(0, dwW-1);
  if (rndH + rndRI >= dwH)
  {
    rndRI = dwH - rndH - 1;
  }
  if (rndW + rndCI >= dwW)
  {
    rndCI = dwW - rndW - 1;
  }
  
  for (int dwRI = 0; dwRI < rndH; dwRI++)
  {
    memset(matIn.data + ((dwRI + rndRI) * dwW + rndCI) * 3, 0, rndW * 3);
  }

  return 0;
}


//pdwRect: x, y, w, h
int mask_plate(cv::Mat &matIn, int *pdwRect)
{
  assert(matIn.type()==CV_8UC3);
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  int dwPX = pdwRect[0], dwPY = pdwRect[1], dwPW = pdwRect[2], dwPH = pdwRect[3];
 
  for (int dwRI = 0; dwRI < dwPH; dwRI++)
  {
    memset(matIn.data + ((dwRI + dwPY) * dwW + dwPX) * 3, 0, dwPW * 3);
  }

  return 0;
}

////////////
void do_augment_plate_onethread(void *p);
extern "C" int do_augment_plate_threads(char *pfns[], int *pdwPlates, int num, 
                                  int stdH, int stdW, float randv, float *pfImgOut)
{
  const int cdwMaxTNum = 48;
  static dg::ThreadPool *psPool = NULL;
  if (psPool == NULL)
  {
    printf("Max Thread Number:%d\n", cdwMaxTNum);
    psPool = new dg::ThreadPool(cdwMaxTNum);
  }

  srand(static_cast<unsigned>(time(0)));

  vector<string> vecfn;
  for (int i = 0; i < num; i++)
  {
    vecfn.push_back(pfns[i]);
  }
  condition_variable cv;
  mutex countmt;
  int dwFinishCount = 0;
  int fnum = vecfn.size();
  Aug_Params *pParams = new Aug_Params[fnum];

  for (int fi=0; fi < fnum; fi++) 
  {
    string strfn = vecfn[fi];
    pParams[fi].strfn = strfn;
    pParams[fi].p_cv = &cv;
    pParams[fi].p_countmt = &countmt;
    pParams[fi].p_FinishCount = &dwFinishCount;
    pParams[fi].needNum = fnum;
    pParams[fi].stdsize[0] = stdH;
    pParams[fi].stdsize[1] = stdW;
    pParams[fi].randv = randv;
    pParams[fi].pfImgOut = pfImgOut + fi * stdH * stdW * 3;
    pParams[fi].pdwRect = pdwPlates + fi * 4;
    psPool->enqueue(do_augment_plate_onethread, (void*)&pParams[fi]);
  }

  unique_lock<mutex> waitlc(countmt);
  cv.wait(waitlc, [&dwFinishCount, &fnum](){return dwFinishCount==fnum;});

//  for (int fi = 0; fi < fnum; fi++)
//  {
//    cv::Mat &img = pParams[fi].matOut;
//    memcpy(pfImgOut + fi * stdH * stdW * 3,  
//           img.data, sizeof(float) * stdH * stdW * 3);
////    cv::imshow("hi", img);
////    cv::waitKey(0);
//  }

  delete []pParams;
  return 0;
}


void do_augment_plate_onethread(void *p)
{
  Aug_Params *pParam = (Aug_Params*)p;
  string &strfn = pParam->strfn; 
  mutex *p_countmt = pParam->p_countmt;
  condition_variable *p_cv = pParam->p_cv;
  int *p_FinishCount = pParam->p_FinishCount;
  int needNum = pParam->needNum;
  int stdH = pParam->stdsize[0];
  int stdW = pParam->stdsize[1];
  float randv = pParam->randv;
  cv::Mat &matOut = pParam->matOut;
  float *pfImgOut = pParam->pfImgOut;
  int *pdwRect = pParam->pdwRect;

  cv::Mat img = cv::imread(strfn);
  if (img.cols==0)
  {
    printf("Can not read image %s\n", strfn.c_str());

    unique_lock<mutex> countlc(*p_countmt);
    if (*p_FinishCount < needNum)
    {
      (*p_FinishCount)++;
    }
    if ((*p_FinishCount) == needNum)
    {
      p_cv->notify_all();
    } 
    countlc.unlock();

    return;
  }
  int dwDoAug = 1;

  if (dwDoAug)
  { 
    //random mask plate
    int rnd0 = rand();
    if (rnd0 > (RAND_MAX / 4))
    {
      mask_plate(img, pdwRect);
    }
  //  printf("%s\n", strfn.c_str()); 
    //mask rows
  //  int rnd0 = rand();
  //  if (rnd0 < (RAND_MAX / 4) * 3)
    {
//      rnd_mask(img);
      rnd_block_mask(img);
    }
    
    //crop
    rnd_crop(img);
  }
  //reisze
  cv::resize(img, img, cv::Size(stdW, stdH));
  //normalize
  normalize_img(img);
  if (dwDoAug)
  {
    //rotate
    rnd_rotate(img, randv);
    //flip
    int rnd = rand();
    if (rnd < RAND_MAX / 2)
    {
      cv::flip(img, img, 1);
    }
  }
//  img.copyTo(matOut);
  float *pfImg = (float*)img.data;

  float *pfOutR = pfImgOut;
  float *pfOutG = pfImgOut + stdH * stdW;
  float *pfOutB = pfImgOut + stdH * stdW * 2;
  for (int ri = 0; ri < stdH; ri++)
  {
    for (int ci = 0; ci < stdW; ci++)
    {
       int dwOft = ri * stdW + ci;
       pfOutR[dwOft] = pfImg[dwOft * 3 + 0];
       pfOutG[dwOft] = pfImg[dwOft * 3 + 1];
       pfOutB[dwOft] = pfImg[dwOft * 3 + 2];
    }
  }

  unique_lock<mutex> countlc(*p_countmt);
  if (*p_FinishCount < needNum)
  {
    (*p_FinishCount)++;
  }
  if ((*p_FinishCount) == needNum)
  {
    p_cv->notify_all();
  } 
  countlc.unlock();

  return;
}

/////////////
void do_augment_plate_part_onethread(void *p);
extern "C" int do_augment_plate_part_threads(char *pfns[], int *pdwPlates, int *pcrop, int num, 
                                  int stdH, int stdW, float randv, float *pfImgOut)
{
  const int cdwMaxTNum = 48;
  static dg::ThreadPool *psPool = NULL;
  if (psPool == NULL)
  {
    printf("Max Thread Number:%d\n", cdwMaxTNum);
    psPool = new dg::ThreadPool(cdwMaxTNum);
  }

  srand(static_cast<unsigned>(time(0)));

  vector<string> vecfn;
  for (int i = 0; i < num; i++)
  {
    vecfn.push_back(pfns[i]);
  }
  condition_variable cv;
  mutex countmt;
  int dwFinishCount = 0;
  int fnum = vecfn.size();
  Aug_Params *pParams = new Aug_Params[fnum];

  for (int fi=0; fi < fnum; fi++) 
  {
    string strfn = vecfn[fi];
    pParams[fi].strfn = strfn;
    pParams[fi].p_cv = &cv;
    pParams[fi].p_countmt = &countmt;
    pParams[fi].p_FinishCount = &dwFinishCount;
    pParams[fi].needNum = fnum;
    pParams[fi].stdsize[0] = stdH;
    pParams[fi].stdsize[1] = stdW;
    pParams[fi].randv = randv;
    pParams[fi].pfImgOut = pfImgOut + fi * stdH * stdW * 3;
    pParams[fi].pdwRect = pdwPlates + fi * 4;
    pParams[fi].pcrop = pcrop + fi * 4;
//    printf("hahahahahahahahahahahahha\n");
    psPool->enqueue(do_augment_plate_part_onethread, (void*)&pParams[fi]);
  }

  unique_lock<mutex> waitlc(countmt);
  cv.wait(waitlc, [&dwFinishCount, &fnum](){return dwFinishCount==fnum;});

//  for (int fi = 0; fi < fnum; fi++)
//  {
//    cv::Mat &img = pParams[fi].matOut;
//    memcpy(pfImgOut + fi * stdH * stdW * 3,  
//           img.data, sizeof(float) * stdH * stdW * 3);
////    cv::imshow("hi", img);
////    cv::waitKey(0);
//  }

  delete []pParams;
  return 0;
}


void do_augment_plate_part_onethread(void *p)
{
  Aug_Params *pParam = (Aug_Params*)p;
  string &strfn = pParam->strfn; 
  mutex *p_countmt = pParam->p_countmt;
  condition_variable *p_cv = pParam->p_cv;
  int *p_FinishCount = pParam->p_FinishCount;
  int needNum = pParam->needNum;
  int stdH = pParam->stdsize[0];
  int stdW = pParam->stdsize[1];
  float randv = pParam->randv;
  cv::Mat &matOut = pParam->matOut;
  float *pfImgOut = pParam->pfImgOut;
  int *pdwRect = pParam->pdwRect;
  int *pcrop = pParam->pcrop;

  cv::Mat img = cv::imread(strfn);
  if (img.cols==0)
  {
    printf("Can not read image %s\n", strfn.c_str());

    unique_lock<mutex> countlc(*p_countmt);
    if (*p_FinishCount < needNum)
    {
      (*p_FinishCount)++;
    }
    if ((*p_FinishCount) == needNum)
    {
      p_cv->notify_all();
    } 
    countlc.unlock();

    return;
  }
  int dwDoAug = 1;

  if (dwDoAug)
  { 
    //random mask plate
    int rnd0 = rand();
  //  printf("mask_plate\n");
    if (rnd0 > (RAND_MAX / 4))
    {
      mask_plate(img, pdwRect);
    }
  //  printf("%s\n", strfn.c_str()); 
    //mask rows
   // printf("mask_finished!\n");
  //  int rnd0 = rand();
  //  if (rnd0 < (RAND_MAX / 4) * 3)
   // {
//      rnd_mask(img);
  //    rnd_block_mask(img);
 //   }
    
    //crop
    //rnd_crop(img);
  }
  // resize to (256,256)  
 // printf("resize to (256,256)\n");
  cv::resize(img,img,cv::Size(256,256));
 // printf("crop\n");
  part_crop1(img, pcrop);    
    
  //reisze
  cv::resize(img, img, cv::Size(stdW, stdH));
  //normalize
  normalize_img(img);
  if (dwDoAug)
  {
    //rotate
    rnd_rotate(img, randv);
    //flip
    int rnd = rand();
    if (rnd < RAND_MAX / 2)
    {
      cv::flip(img, img, 1);
    }
  }
//  img.copyTo(matOut);
  float *pfImg = (float*)img.data;

  float *pfOutR = pfImgOut;
  float *pfOutG = pfImgOut + stdH * stdW;
  float *pfOutB = pfImgOut + stdH * stdW * 2;
  for (int ri = 0; ri < stdH; ri++)
  {
    for (int ci = 0; ci < stdW; ci++)
    {
       int dwOft = ri * stdW + ci;
       pfOutR[dwOft] = pfImg[dwOft * 3 + 0];
       pfOutG[dwOft] = pfImg[dwOft * 3 + 1];
       pfOutB[dwOft] = pfImg[dwOft * 3 + 2];
    }
  }

  unique_lock<mutex> countlc(*p_countmt);
  if (*p_FinishCount < needNum)
  {
    (*p_FinishCount)++;
  }
  if ((*p_FinishCount) == needNum)
  {
    p_cv->notify_all();
  } 
  countlc.unlock();

  return;
}
  
////////////
void do_get_test_onethread(void *p);
extern "C" int do_get_test_threads(char *pfns[], int num, 
                                  int stdH, int stdW, float *pfImgOut)
{
  const int cdwMaxTNum = 48;
  static dg::ThreadPool *psPool = NULL;
  if (psPool == NULL)
  {
    printf("Max Thread Number:%d\n", cdwMaxTNum);
    psPool = new dg::ThreadPool(cdwMaxTNum);
  }

  srand(static_cast<unsigned>(time(0)));

  vector<string> vecfn;
  for (int i = 0; i < num; i++)
  {
    vecfn.push_back(pfns[i]);
  }
  condition_variable cv;
  mutex countmt;
  int dwFinishCount = 0;
  int fnum = vecfn.size();
  Aug_Params *pParams = new Aug_Params[fnum];

  for (int fi=0; fi < fnum; fi++) 
  {
    string strfn = vecfn[fi];
    pParams[fi].strfn = strfn;
    pParams[fi].p_cv = &cv;
    pParams[fi].p_countmt = &countmt;
    pParams[fi].p_FinishCount = &dwFinishCount;
    pParams[fi].needNum = fnum;
    pParams[fi].stdsize[0] = stdH;
    pParams[fi].stdsize[1] = stdW;
    pParams[fi].pfImgOut = pfImgOut + fi * stdH * stdW * 3;
    psPool->enqueue(do_get_test_onethread, (void*)&pParams[fi]);
  }

  unique_lock<mutex> waitlc(countmt);
  cv.wait(waitlc, [&dwFinishCount, &fnum](){return dwFinishCount==fnum;});

  delete []pParams;
  return 0;
}


void do_get_test_onethread(void *p)
{
  Aug_Params *pParam = (Aug_Params*)p;
  string &strfn = pParam->strfn; 
  mutex *p_countmt = pParam->p_countmt;
  condition_variable *p_cv = pParam->p_cv;
  int *p_FinishCount = pParam->p_FinishCount;
  int needNum = pParam->needNum;
  int stdH = pParam->stdsize[0];
  int stdW = pParam->stdsize[1];
  cv::Mat &matOut = pParam->matOut;
  float *pfImgOut = pParam->pfImgOut;
  int *pdwRect = pParam->pdwRect;

  cv::Mat img = cv::imread(strfn);
  if (img.cols==0)
  {
    printf("Can not read image %s\n", strfn.c_str());

    unique_lock<mutex> countlc(*p_countmt);
    if (*p_FinishCount < needNum)
    {
      (*p_FinishCount)++;
    }
    if ((*p_FinishCount) == needNum)
    {
      p_cv->notify_all();
    } 
    countlc.unlock();

    return;
  }

  //reisze
  cv::resize(img, img, cv::Size(stdW, stdH));
  //normalize
  normalize_img(img);

  float *pfImg = (float*)img.data;

  float *pfOutR = pfImgOut;
  float *pfOutG = pfImgOut + stdH * stdW;
  float *pfOutB = pfImgOut + stdH * stdW * 2;
  for (int ri = 0; ri < stdH; ri++)
  {
    for (int ci = 0; ci < stdW; ci++)
    {
       int dwOft = ri * stdW + ci;
       pfOutR[dwOft] = pfImg[dwOft * 3 + 0];
       pfOutG[dwOft] = pfImg[dwOft * 3 + 1];
       pfOutB[dwOft] = pfImg[dwOft * 3 + 2];
    }
  }

  unique_lock<mutex> countlc(*p_countmt);
  if (*p_FinishCount < needNum)
  {
    (*p_FinishCount)++;
  }
  if ((*p_FinishCount) == needNum)
  {
    p_cv->notify_all();
  } 
  countlc.unlock();

  return;
}



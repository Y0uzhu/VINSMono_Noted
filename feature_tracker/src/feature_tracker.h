#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;//鱼眼专用，去除边缘噪声的，因为鱼眼边缘畸变太大
    cv::Mat prev_img, cur_img, forw_img;//对应帧图像，这个以时间来论，forw>cur>prev，即forw最新
    vector<cv::Point2f> n_pts;//向量，其中每个元素都是通过特征提取获得的特征点（而其余特征都是通过光流得到的）
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//对应图像帧中的特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;//归一化相机坐标系下的坐标
    vector<cv::Point2f> pts_velocity;//cur帧相对prev帧特征点沿x,y方向的像素移动速度
    vector<int> ids;//被追踪到的特征点的ID
    vector<int> track_cnt;//每个特征点被追踪到的次数
    map<int, cv::Point2f> cur_un_pts_map;//构建id与归一化坐标的id，见undistortedPoints()
    map<int, cv::Point2f> prev_un_pts_map;//与cur_un_pts_map相同，但是prev帧的
    camodocal::CameraPtr m_camera;//相机模型
    double cur_time;//cur帧时间
    double prev_time;//prev帧时间

    static int n_id;//ID，用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
};

#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;//图片的高
extern int COL;//图片的宽
extern int FOCAL_LENGTH;//这个是默认的460
const int NUM_OF_CAM = 1;//相机数量


extern std::string IMAGE_TOPIC;//图片话题名
extern std::string IMU_TOPIC;//IMU话题名
extern std::string FISHEYE_MASK;//这个是鱼眼掩膜的文件路径，得看是否是鱼眼
extern std::vector<std::string> CAM_NAMES;//这个不重要就是camera，双目的话应该是不一样的
extern int MAX_CNT;//这个最大特征点数
extern int MIN_DIST;//特征点的最小距离
extern int WINDOW_SIZE;//滑窗大小，这个默认是20，在视觉惯性对齐里面用到的
extern int FREQ;//话题发布频率
extern double F_THRESHOLD;//随机采样的阈值
extern int SHOW_TRACK;//是否发布跟踪的图片
extern int STEREO_TRACK;//是否为双目跟踪
extern int EQUALIZE;//是否直方图均衡
extern int FISHEYE;//是不是鱼眼
extern bool PUB_THIS_FRAME;//是否发布这一帧

void readParameters(ros::NodeHandle &n);

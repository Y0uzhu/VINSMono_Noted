#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame//图像帧
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        // 这是一个与图像对应的map，map的每一个元素代表一个特征点，map.first代表特征点的id，map.second是一个vector。
        // 该vector中每个元素相当于一个相机，如果只有一个相机，那么该vector只有一个元素，即map.second[0]，这是一个pair。
        // 该pair.first是相机的id，pair.second是一个7*1的矩阵，代表在该帧下该相机的一个特征点
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points; 
        double t;// 图像帧的时间戳
        Matrix3d R; // 假设当前帧为第i帧，则表示R_cl_bi
        Vector3d T;// T_cl_ci
        IntegrationBase *pre_integration;//该帧的与积分值，包括pvq状态和雅可比、协方差矩阵
        bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);
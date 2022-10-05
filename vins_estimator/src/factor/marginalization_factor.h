#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;
// 这里将参数块分为Xm,Xb,Xr。Xm表示被marg掉的参数块，Xb表示与Xm相连接的参数块，Xr表示剩余的参数块。
// 将参数块Xm与Xb之间的约束记为Zb，Xb内部之间的约束记为Zc，剩余的约束记为Zr。

struct ResidualBlockInfo// 这个类保存了待marg变量（Xm）与相关联变量（Xb）之间的一个约束因子关系  -  Zm
{
    /**
     * @brief 构造函数，传入代价函数，损失函数，输入参数和待marg的优化变量id
     * 
     * @param _cost_function 
     * @param _loss_function 
     * @param _parameter_blocks 
     * @param _drop_set 
     */
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();//调用cost_function的evaluate函数计算残差 和 雅克比矩阵

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    // 在构建ResidualBlockInfo对象时传入的变量数据，不同信息的数据不同，比如imu的是[7,9,7,9]，视觉是[7,7,7,1]，
    // 先验信息我还没弄明白是啥，但应该是各滑窗帧的位姿和vb[7,9]*10
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;// 待marg的优化变量id，imu约束中输入的为{0, 1}
    // 雅可比
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;// 残差，imu15*1，视觉2*1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo// 保存了优化时上一步边缘化后保留下来的先验信息
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();// 得到每次IMU和视觉观测对应的参数块，雅克比矩阵，残差值
    void marginalize();// 开启多线程构建信息矩阵H和b ，同时从H,b中恢复出线性化雅克比和残差
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    // xm与xb之间所有的约束因子，Zb
    std::vector<ResidualBlockInfo *> factors;
    // m所有将被marg掉变量的localsize之和，如上面的Xm的localsize，n为所有与将被marg掉变量有约束关系的变量的localsize之和，如上面的Xb的localsize
    int m, n;
    // 信息矩阵中的所有的参数块信息，包括xm和xb，储存格式为：<参数块地址，参数块的global size>
    std::unordered_map<long, int> parameter_block_size; 
    int sum_block_size;
    // 排序后的信息矩阵的参数块信息，待marg的参数块在前，其余在后，存储格式为：<参数块地址，参数块排序好后的索引>
    std::unordered_map<long, int> parameter_block_idx; 
    // <参数块地址，参数块数据>，需要注意的是这里保存的参数块数据是原始参数块数据的一个拷贝，不再变化，用于记录这些参数块变量在marg时的状态
    std::unordered_map<long, double *> parameter_block_data;
    // 上一帧没被marg的参数块<参数块地址，参数块的globalsize>
    std::vector<int> keep_block_size;
    // 上一帧没被marg的参数块<参数块地址，参数块的索引>，这个索引是从m开始的
    std::vector<int> keep_block_idx;
    // 上一帧没被marg的参数块拷贝
    std::vector<double *> keep_block_data;
    // 线性化的雅可比和残差
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;// 这个传入时是last_marginalization_info量
};

#include "feature_manager.h"
/**
 * @brief 
 * 
 * @return int 刚发现时的帧索引+被观测到的帧数-1（减1是因为被观测到的帧中包含发现帧）
 */
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}
/**
 * @brief 设置外参
 * 
 * @param _ric 
 */
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}
/**
 * @brief 返回参与优化的路标点的数量（路标至少被观测到两次且路标点第一次发现的时候较早？不理解）
 * 
 * @return int 
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();
        // 路标至少被观测到两次且路标点较早？
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief 向路标点的list中添加路标点，同时判断当前图像帧是否为关键帧，有三个判断条件可以返回true：
 *                1、该帧是起始的几帧或者该帧中的特征点与已有的路标关联成功的特征数量较少
 *                2、该帧没有较远地（路标点第一次出现的帧数比当前帧早很多）可以一直被连续观测到的路标点
 *                3、虽然有这种路标点，但该路标点最近的两个特征之间的距离差比较大
 * 
 * @param frame_count 滑窗帧的索引
 * @param image 图像帧的所有特征信息
 * @param td imu与相机数据间由硬件引起的时间误差
 * @return true 
 * @return false 
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    //遍历当前帧的每一个特征点，将其加入到路标点的list中
    for (auto &id_pts : image)
    {
        //拿出第一个相机的一个特征点，构建一个FeaturePerFrame
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
        // 特征点的id
        int feature_id = id_pts.first;
        // 在所有已有的路标点中，查找该特征点，如果找到了，就返回该路标的迭代器，没有找到，则会返回feature.end()
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) { return it.feature_id == feature_id; });
        //没有找到，就新建一个路标点，并将这个特征点加入到其feature_per_frame中
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        //找到了就将这个特征点加入到其feature_per_frame中
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }
    // 滑窗中帧数非常少或者与已有的路标关联成功的特征数量非常少，则认为是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;
    // 遍历每一个路标，计算路标在较晚发现的两帧中的特征点的距离差
    for (auto &it_per_id : feature)
    {
        // 路标第一次发现时至少是在两帧之前，并且到当前帧为止，最多只有一帧没有发现该路标
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;// 计算距离差的均值，大于阈值则返回true
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}
/**
 * @brief 两帧的特征点匹配
 * 
 * @param frame_count_l l帧
 * @param frame_count_r r帧
 * @return vector<pair<Vector3d, Vector3d>> 两帧中所有的匹配特征点组成匹配对，first是l帧
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // 该观测点要求在l帧之前发现，且到r帧为止都能被连续观测到（如果不这样定义的话，无法定位特征是在哪个帧中出现的）
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}
/**
 * @brief 将所有特征点的深度设为-1
 * 
 * @param x 
 */
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}
/**
 * @brief 路标点深度的倒数，逆深度
 * 
 * @return VectorXd 每一个元素对应一个路标的深度的倒数
 */
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}
/**
 * @brief 根据当前的位姿三角化特征点，得到路标在其发现帧下的深度值，原理详见https://zhuanlan.zhihu.com/p/55530787
 * 
 * @param Ps 
 * @param tic 
 * @param ric 
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历每一个路标
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 保证路标至少是出现过两次的
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        // 将路标首次被发现的认为i帧，i的上一帧认为是j帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // 这个Rs是从哪传过来的，假设他代表R_cl_bi，那么R0就表示R_cl_ci
        // T_w_ci
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
        // 遍历路标的每个特征，构建三角化的系数矩阵，详见https://zhuanlan.zhihu.com/p/55530787
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // T_w_cj
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];//R1就表示R_cl_cj
            // (T_w_ci)T * (T_w_cj) = T_ci_cj
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;//R_ci_cj
            // T_ci_cj的逆，T_cj_ci
            Eigen::Matrix<double, 3, 4> P;//T_cj_ci
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // 使用SVD求最小二乘解
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // 求出来的可能不满足齐次坐标形式（第四个元素为1），所以除以第四个项
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}
/**
 * @brief 把首次在原先最老帧出现的特征点转移到原先第二老帧的相机坐标里
 * 
 * @param marg_R 滑窗原先的最老帧(被merge掉)的旋转(c->w)
 * @param marg_P 滑窗原先的最老帧(被merge掉)的平移(c->w)
 * @param new_R 滑窗原先第二老的帧(现在是最老帧)的旋转(c->w)
 * @param new_P 滑窗原先第二老的帧(现在是最老帧)的平移(c->w)
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 遍历每一个路标点
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 路标的起始帧不为0（不是在最老帧中发现的），则--
        if (it->start_frame != 0)
            it->start_frame--;
        else// 如果是在最老帧中发现的
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());// 把最老帧的特征去掉
            // 如果此时只能被一帧观测到，那就没用了，删掉这个路标
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;// 获得3D坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;// 转换到世界系
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);// 转换到第二老帧
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}
/**
 * @brief 拿出该路标较晚发现的两个特征点，并返回这俩路标点的距离差
 * 
 * @param it_per_id 一个路标点
 * @param frame_count 当前帧id
 * @return double 
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    // 拿出该路标较晚出现的两个特征点
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    // 这是在干嘛？p_i(2)不确定是1吗？dep_i_comp和p_i不是一样的吗？
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
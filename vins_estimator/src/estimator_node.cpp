#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;//imu信息队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf;//特征信息队列
queue<sensor_msgs::PointCloudConstPtr> relo_buf;//重定位信息队列
int sum_of_wait = 0;
//锁
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
// 这部分的tmp_X量究竟是干什么的，他会在最开始时通过积分得到一个tmp_X，
Eigen::Vector3d tmp_P;// p_w_b
Eigen::Quaterniond tmp_Q;// q_w_b
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;//上一帧imu的加速度
Eigen::Vector3d gyr_0;//上一帧imu的角速度
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
/**
 * @brief 通过简单的预积分，更新tmp_X的值.使用上一时刻的姿态进行快速的imu预积分，用来预测最新P,V,Q的姿态，
 * tmp_P,tmp_Q,tmp_V,acc_0,gyr_0 最新时刻的姿态。 这个的作用是为了刷新姿态的输出，但是这个值的误差相对会比较大，
 * 因为虽然这些tmp姿态会通过优化更新，但是优化更新的频率要比imu发布的频率低很多。
 * 
 * @param imu_msg 
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    //判断是否是第一帧imu_msg
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;
    //imu在自己坐标系下的加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};
    //角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
    //上一帧imu在世界坐标系下的加速度
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;
    //imu的角速度
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    //这一帧imu在世界坐标系下的加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    //两帧加速度取中值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    //pvq方程的离散形式
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}
/**
 * @brief 依据滑窗帧的状态更新当前所有的imu帧的pvq，但是
 * 
 */
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;
    // 这里为何还要对所有的imu帧再作积分预测呢，难道估计器滑窗中的最后一帧对应的是上一个图像帧？
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}
/**
 * @brief 完成imu与特征的时间戳对其，并将对齐的imu数据与特征放入到measurements去，如果没能对齐，则返回一个空的。
 * 
 * @return std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> imu数据能够完全包含整个的数据帧，
 * 从比上一帧早一点儿开始，到比当前帧晚一点儿结束
 */
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        // 时间戳对齐
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        //取出最前面一阵的特征
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();
        // 这里只判断了imu时间要早于特征的时间，所以在第一帧时可能会出现多余的imu帧，其余基本刚好能够包含两帧特征跨度的imu数据
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 这里将刚好在图像帧之后的一帧imu数据也加入了，并且没有再删除，这保证了下一个图像帧imu数据的完整，却也会出现imu重叠的现象
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}
/**
 * @brief 将传入的imu数据加入buf中，并通过简单的积分得到最后传入的imu帧的tmp_X状态值（是每次传入的imu都会作积分，所以呈现出的结果就是最近的imu帧的）
 * 
 * @param imu_msg 
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();// 目的是唤醒队列中的第一个线程

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        // 模板类std::lock_guard，在构造时就能提供已锁的互斥量，并在析构的时候进行解锁,在std::lock_guard对象构造时，
        // 传入的mutex对象(即它所管理的mutex对象)会被当前线程锁住
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//获取临时的P,V,Q值
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
         //如果solver_flag为非线性优化，则根据imu预积分的pvq发布最新的里程计消息
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

/**
 * @brief 将特征信息放入buf中
 * 
 * @param feature_msg 
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}
/**
 * @brief 当接收到重置消息（该消息会在图像数据的时间戳出错时发布true）时，删除最近帧的特征和imu数据；同时清除估计器的状态和参数，时间重置
 * 
 * @param restart_msg 
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}
/**
 * @brief 将重定位消息放入到buf中，该消息由pose_graph发布而来，这个消息是由后端发来的，其包含points和channels两种数据
 * points中存储回环帧中特征点的归一化坐标（x，y）和路标点的id（z）
 * channels共8维，存储了对应的回环帧的位姿（位移0-2，旋转3-6）以及帧索引（7）
 * 
 * @param points_msg 
 */
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        // imu与图像帧组成的配对儿，imu数据能够完全包含整个的数据帧， 从比上一帧早一点儿开始，到比当前帧晚一点儿结束
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        // 用了一个lambda表达式，也就是说，在return里面的部分是false时，保持上锁状态，继续配对数据；
        // 如果return里面时true，说明配对完成，释放锁，measurements完成了，以供后续使用。
        con.wait(lk, [&] { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            /*******************************************************************************************/
            // 对imu信息的处理，向估计器中传递预积分数据，计算预积分项、雅可比和协方差等。并且获得了滑窗中pvq各状态量的初值
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    //先做插值
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            /*******************************************************************************************/
            // 对重定位信息的处理，抽出一帧重定位数据传入估计器中
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                // points中存储回环帧中特征点的归一化坐标（x，y）和路标点的id（z）
                // channels共8维，存储了对应的回环帧的位姿（位移0-2，旋转3-6）以及帧索引（7）
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            /*******************************************************************************************/
            // 对图像特征的处理
            // 一帧图像所包含的所有数据都在这个img_msg中，其包括points项和channels项。
            // points项传递cur帧所有特征点的归一化坐标，channels项则包含许多。
            // channels[0]传递特征点的id序列，channels[1]和channels[2]传递cur帧特征点的像素坐标序列，
            // channels[3]和channels[4]传递cur帧相对prev帧特征点沿x,y方向的像素移动速度
            TicToc t_s;
            // 这是一个与图像对应的map，map的每一个元素代表一个特征点，map.first代表特征点的id，map.second是一个vector。
            // 该vector中每个元素相当于一个相机，如果只有一个相机，那么该vector只有一个元素，即map.second[0]，这是一个pair。
            // 该pair.first是相机的id，pair.second是一个7*1的矩阵，代表在该帧下该相机的一个特征点
            // 该特征点从左到右依次为：特征的归一化坐标[3]、特征的像素坐标[2]、特征的移动速度[2]
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // 获得特征的id和相机id
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;//从左到右依次为：特征的归一化坐标[3]、特征的像素坐标[2]、特征的移动速度[2]
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        // 在进行完一个process()循环后，当前的PVQ的状态和循环开始的状态是不一样的。所以说我们需要再根据当前的数据，更新当前的PVQ状态.
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}

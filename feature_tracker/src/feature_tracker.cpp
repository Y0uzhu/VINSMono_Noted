#include "feature_tracker.h"

int FeatureTracker::n_id = 0;//ID，用来作为特征点id，每检测到一个新的特征点，就将++n_id作为该特征点
/**
 * @brief 检查点是否在图像边界内
 * 
 * @param pt 
 * @return true 
 * @return false 
 */
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}
/**
 * @brief 按照状态量来重新修改特征点，只保留了那些状态为1的特征点
 * 
 * @param v 
 * @param status 
 */
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
/**
 * @brief 按照状态量来重新修改id和追踪的次数等，只保留了那些状态为1的
 * 
 * @param v 
 * @param status 
 */
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}
/**
 * @brief 删除了当前帧中距离过近的特征点，目的为了特征点的均匀化，更新了均匀化后的特征点、特征点ID和特征点追踪到的次数
 *                同时给现有的特征点设置mask，mask与图像大小相同，但已提取过特征的区域置0，而其余区域为255
 * 
 */
void FeatureTracker::setMask()
{
    // 判断是不是鱼眼相机，构建一个mask矩阵，像素初始值设为255
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    //这个量的结构大致为vector<pair<特征点被追踪到的次数, pair<特征点坐标, 特征点的ID>>>
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    // 按照特征点被追踪到的次数从大到小排序，因为根据光流特点，追踪多的稳定性好
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    //清空已有的三项数据
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // 判断每个特征点的位置在mask上的值是否为255，是的话则将该特征点的数据重新加入，并将mask中该点及其周围点的值置为0，
        // 这就保证了同一区域不会存在大量聚集的特征点
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}
/**
 * @brief 把通过特征点提取获得的特征加入到原有的forw帧特征点、特征点ID和特征点追踪到的次数中
 * 
 */
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        // 由于这是通过特征提取得到的特征，所以把id置为-1，在更新id时再获得新的id
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}
/**
 * @brief 通过光流法获得初始的特征点，如果数量不够，则再通过特征提取补全，最后还计算了prev帧到cur帧特征的移动速度。
 *                需要注意的是，光流法无论是不是发布帧都会提取，而特征提取则只在发布帧进行
 * 
 * @param _img 输入的图像
 * @param _cur_time 输入图像的时间戳
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;
    /*******************************************************************************************/
    //增强输入图像的对比度，也叫均衡化（图像太暗或者太亮，提特征点比较难，所以均衡化一下）
    if (EQUALIZE)
    {
        // CLAHE是另外一种直方图均衡算法，能有效的增强或改善图像（局部）对比度，从而获取更多图像相关边缘信息有利于分割，两个参数分别为阈值和尺寸
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        // 均衡化运行的函数，第一个参数为输入，第二个为输出
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;
    /*******************************************************************************************/
    // 更新图像，这里forw表示当前，cur表示上一帧
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    /*******************************************************************************************/
    // 光流法获取特征
    forw_pts.clear();

    if (cur_pts.size() > 0)// cur帧有特征点，就可以进行光流追踪了
    {
        TicToc t_o;
        vector<uchar> status;//输出状态向量（无符号字符）;如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0
        vector<float> err;//输出错误的矢量; 向量的每个元素都设置为相应特征的错误，错误度量的类型可以在flags参数中设置; 如果未找到流，则未定义错误（使用status参数查找此类情况）。
        //利用opencv进行了光流追踪，并得到了特征点及其状态
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        // 根据特征是否在边界内更新status，再根据status更新各状态量
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))//状态量置一且点不在边界内时，状态位置零
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status); // 去畸变后的坐标
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    // 被追踪到的是上一帧就存在的，因此追踪数+1
    for (auto &n : track_cnt)
        n++;
    /*******************************************************************************************/
    // 如果这是发布帧，才继续下面的操作
    if (PUB_THIS_FRAME)
    {
        rejectWithF();//通过对级约束来剔除outlier
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//将forw帧的特征点均匀化
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        /*******************************************************************************************/
        //这里就是说你规定要检测MAX_CNT个特征点，现在追踪到了forw_pts.size()个，所以你就需要一些新检测的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            //从mask剩余为255的区域（即未提取特征的区域）中提取角点
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();//如果特征足够，就把n_pts清空，因为下面要把n_pts中的点加入到forw_pts中，所以不需要就得清空
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}
/**
 * @brief 通过对级约束来剔除outlier，先将两帧的特征点变换到同一个虚拟相机的像素坐标系下，
 * 再调用cv中计算两帧本质矩阵的函数来更新status，在根据status更新各项数据
 * 
 */
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // 这里用一个虚拟相机，将特征点先变换到归一化坐标系下，再变换到一个虚拟相机的像素坐标系下，因此该部分目的是得到在虚拟相机上两帧的光流追踪点
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // 得到相机归一化坐标系的值
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // opencv接口计算本质矩阵，某种意义也是一种对级约束的outlier剔除，
        // 这里在根本上好像是利用该计算对status矩阵进行了进一步的outlier剔除
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        //依据状态量来对各项数据进行更新
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}
/**
 * @brief 将原本id赋为-1的特征修改为新的id，而新的id则通过不断增长的n_id获得
 * 
 * @param i 
 * @return true 
 * @return false 
 */
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
/**
 * @brief 该函数并非实际意义上的去畸变函数，而是求得了pts_velocity量，其为在归一化坐标系中cur帧相对prev帧特征点沿x,y方向的移动速度
 *                是通过计算cur和prev帧归一化坐标的差值得到的，如果某特征在prev帧没有发现，则该特征的移动速度为0
 * 
 */
void FeatureTracker::undistortedPoints()
{
    //获得cur帧的归一化坐标和与id对应的归一化坐标
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)// -1为特征追踪获得的特征，不计算在内
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

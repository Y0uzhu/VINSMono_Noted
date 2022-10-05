#include "marginalization_factor.h"
/**
 * @brief 将不同的costfunction在此处进行整合，规定残差和雅可比的大小，并在这里调用各自的Evaluate函数计算残差和雅可比
 * 最后还为视觉部分添加了核函数
 * 
 */
void ResidualBlockInfo::Evaluate()
{
    // 获得约束各自的残差个数，在构建cost_function时，后面中括号中的第一项即残差的维度，所以num_residuals()返回的应该就是中括号中的第一项的大小
    residuals.resize(cost_function->num_residuals());
    // cost_function中参数块的大小
    // 在构建cost_function时，后面中括号中的第一项是残差的维度，而后面的则是参数块的维度，
    // 所以parameter_block_sizes()返回的应该就是中括号中的第一项后面的几项的大小
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    //数组外围的大小，也就是参数块的个数
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());
    //分配每一行的大小，残差的维数*每个参数块中参数的个数block_sizes[i]，J矩阵大小的确认！想一下
    //比如：两个残差f1,f2;5个变量x1,x2,,,x5, 则J矩阵是2行5列呀
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    //利用各自残差的Evaluate函数计算残差和雅克比矩阵。
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);
    //好像明白了，这个是视觉里面有的Huber核函数；有鲁棒核函数的残差部分，重写雅克比与残差
    if (loss_function)// 因为先验和imu输入的损失函数都是null，所以这里只有视觉部分可以通过
    {
        double residual_scaling_, alpha_sq_norm_;
        // 鲁棒核函数的输入与导数
        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);// 一阶导的开平方

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }
        //感觉像根据先验残差，推出新的残差和雅可比公式一样！
        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}
/**
 * @brief 根据输入的约束信息传入信息矩阵的参数块parameter_block_size和待marg的参数块parameter_block_idx。
 * 但是待marg的参数块中仅仅有待marg参数块的内存地址，而其对应的id全部为；
 * 
 * @param residual_block_info 约束信息
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);// 将约束信息添加到总的约束因子中去
    // 约束的输入变量
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // 构建的代价函数中的参数块大小，
    // 在构建cost_function时，后面中括号中的第一项是残差的维度，而后面的则是参数块的维度，
    // 所以parameter_block_sizes()返回的应该就是中括号中的第一项后面的几项的大小
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();
    // 这一步好像是在传入信息矩阵的参数块，用输入变量的地址和代价函数中的参数块大小来填充信息矩阵的参数块
    // 但是，输入变量的参数块大小与代价函数中的参数块大小相同，为什么不能直接使用输入变量的参数块呢
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];//指向数据的指针
        int size = parameter_block_sizes[i];//因为仅仅有地址不行，还需要有地址指向的这个数据的长度
        parameter_block_size[reinterpret_cast<long>(addr)] = size;//将优化的变量地址所对应的size改为parameter_block_sizes[i]
    }
    //找到待marg的参数块的地址，但并没有赋id
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}
/**
 * @brief  计算所有约束的残差和雅可比，并记录了参数块数据的拷贝信息parameter_block_data
 * 
 */
void MarginalizationInfo::preMarginalize()
{
     //遍历所有factor，在前面的addResidualBlockInfo中会将不同的残差块加入到factor中。
    for (auto it : factors)
    {
        it->Evaluate();//计算所有约束的残差和雅克比矩阵
        // cost_function中参数块的大小
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        //给parameter_block_data填充值
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}
/**
 * @brief 将7维的位姿转换为6维
 * 
 * @param size 
 * @return int 
 */
int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}
/**
 * @brief 让大小强制为7
 * 
 * @param size 
 * @return int 
 */
int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}
/**
 * @brief 构建整体矩阵所有的A和b
 * 
 * @param threadsstruct 
 * @return void* 
 */
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    //遍历该线程分配的所有factors，所有观测项
    for (auto it : p->sub_factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            //得到参数块的大小
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)//对于pose来说，是7维的,最后一维为0，这里取左边6
                size_i = 6;
            //只提取local size部分，对于pose来说，是7维的,最后一维为0，这里取左边6维
            //P.leftCols(cols) = P(:, 1:cols)，取出从1列开始的cols列
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)//对应对角区域，H*X=b, A代表H矩阵
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else//对应非对角区域
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            //求取g，Hx=g，都是根据公式来写程序的！
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}
/**
 * @brief 开启多线程构建信息矩阵H和b，同时从H,b中恢复出舒尔补的线性化雅克比和残差
 * 
 */
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    // 这里为parameter_block_idx填充了索引，将参数块堆叠，用不断叠加的参数块的localsize来作为索引，并通过参数块的地址在parameter_block_size寻找对应参数块的大小
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    m = pos;//需要marg掉的变量个数
    // 注意，前面的parameter_block_idx中一直是只有待marg的参数块，下面是将其余的参数块添加到后面
    for (const auto &it : parameter_block_size)
    {
         //遍历每一个参数块，如果这个变量不是是待marg的优化变量
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            // 新插入一个参数块，并为其设置索引
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second); //索引叠加
        }
    }
    //上面的操作是在原有的m个上又加的，所以要保留下来的变量个数n要再-m
    n = pos - m;

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);//整个矩阵大小 --- 没有边缘化之前的矩阵
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];//携带每个线程的输入输出信息
    //为每个线程均匀分配factor
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    //构造4个线程，并确定线程的主程序
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));//分别构造矩阵
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    //将每个线程构建的A和b加起来
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        pthread_join( tids[i], NULL ); //阻塞等待线程完成
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    //TODO
    /*代码中求Amm的逆矩阵时，为了保证数值稳定性，做了Amm=1/2*(Amm+Amm^T)的运算，Amm本身是一个对称矩阵，所以  等式成立。
    接着对Amm进行了特征值分解,再求逆，更加的快速*/
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());
    //舒尔补，上面这段代码边缘化掉xm变量，保留xb变量
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);//0，m是开始的位置，m,m是开始位置后的大小
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;//这里的A和b是marg过的A和b,大小是发生了变化的
    //下面就是更新先验残差项
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);//求更新后 A特征值
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
    //求取特征值及其逆的均方根
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    //分别指的是边缘化之后从信息矩阵A和b中恢复出来雅克比矩阵和残差向量；
    //两者会作为先验残差带入到下一轮的先验残差的雅克比和残差的计算当中去
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}
/**
 * @brief 填充对象中的keep_block部分，即上一帧没被marg的参数块
 * 
 * @param addr_shift 
 * @return std::vector<double *> 在MargOld部分，返回的参数块中为0-9帧。在MargSecNew部分则是保留了0-9帧
 */
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        // 大于m就是只留下了没有marg的参数块
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}
/**
 * @brief 先验代价函数的构造函数。与其他的构造函数不同，正常的代价函数在构造时要在中括号里规定残差的维度、输入数据的维度。
 * 但是由于先验约束的残差和输入的维度不确定，所以通过set_num_residuals()和mutable_parameter_block_sizes()来自动输入残差和数据的维度
 * 
 * @param _marginalization_info 
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};
/**
 * @brief 
 * 
 * @param parameters 上一个优化帧的滑窗中的0-9帧的pqvb信息
 * @param residuals 
 * @param jacobians 
 * @return true 
 * @return false 
 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}

    
    int n = marginalization_info->n;// 上一帧没有被marg掉的数量
    int m = marginalization_info->m;// 上一帧被marg掉的数量
    Eigen::VectorXd dx(n);
    // 遍历上一帧所有没被marg掉的参数块
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        // 得到该参数块的大小和序号（减m保证序号是从开始的）
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        // x 是上一帧marg完之后的参数块
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        // x0表示上一帧marg之前的参数块
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        // 这俩参数块相减，得到的是先验约束的x的变化量
        if (size != 7)// speed_bias 9项
            dx.segment(idx, size) = x - x0;//变量更新，全是向量，可以直接加减
        else// 位置，姿态项, 姿态不能直接加减了！要变成四元素的 ✖
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();// 位置可以直接加减
            // 姿态用的四元数的加减
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // 计算残差，用泰勒展开求的公式，但marginalization_info->linearized_residuals和linearized_jacobians怎么来的现在还不知道。
    // linearized_residuals和linearized_jacobians都是在marg部分通过舒尔补获得的先验约束的雅可比和残差
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    // 雅可比直接根据marginalization_info->linearized_jacobians而来，但这个怎么来的也不知道
    if (jacobians)
    {
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}

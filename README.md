### 模型设计的指导
1. 修改采样的方案，通过每隔几轮的更新候选集合进行采样
    - 采样中当选择了(x_a, x_p)之后，如何确定选择的x_n是一个可以提升结果的点  
2. 细化case方案，重新定制损失函数，把损失函数可视化出来
3. 设计x_a, x_p, x_n之间的矢量信息，求出夹角方向值，重新设计损失函数
4. 通过增大的batch信息，将类内误差和类间误差添加到损失函数中去


### 问题以及解决？
1. 所有的训练样本都是根据随机选择的，其中存在部分数据是很难被直接选择到的，导致10分类的分类器的分类性能下降
2. 改进样本构造的方案，使得所有的样本都可以进入分类器进行训练

### 实验结果
* Triple Loss + Classification + Cosine Random Samples
 <img src="https://github.com/liuguiyangnwpu/MassImageRetrieval/blob/master/experiment/showImages/triple_classifiy.png" width = "400" height = "400" alt="实验结果图" align=center />

### TODOLIST
- [x] 使用Res50提取图像的特征
- [x] 编写孪生网络进行测试
- [x] 编写Triple Loss网络，并进行测试
- [ ] 重新设计Triple Loss网络训练样本的构造
    - [x] 添加了基于聚类中心的anchor选择和在给定半径之外的正负样本的选择
    - [x] 添加了针对训练样本中`$(x_a, x_p, x_n)$`之间的方向条件进行选择
    - [ ] 添加针对Query列表候选集进行训练样本选择的策略
- [ ] 根据TripleModel输入的数据中可以转化成PairWise的排序问题
- [x] 将每次训练出得模型结果保存成文件便于后续分析
- [ ] 结果图中，聚类不够紧凑
    - [ ] 针对数据采样策略的修改
        - [x] 在采样时使用一个set，保证被采样过的样本不能在被采样一次，直到没有可采样数据后，结束这一轮的训练
        - [x] 每一个batch采样时，将记录每个样本被采样的次数，每次会得到一个分布，将分布改成概率p，下一次按照(1-p)去进行采样
        - [ ] 损失函数为`max(0, dist loss)`，在训练段记录为0的样本，这些样本对整体训练没有梯度的贡献，进而指导采样
        - [ ] 每一轮训练后，会得到全量数据的距离矩阵，将距离矩阵转换成概率矩阵对采样端进行结果指导(MCMC)
    - [ ] 修改loss函数策略
        - 没有关注到x_p到x_a的距离的控制
        - 是否可以引入EM算法，对进行二维变量的混合高斯估计
    - [ ] 当选择的数据sample(x_a, x_p, x_n)为一下情况，样本失效(目标是max(0.0, dist_p - dist_n + margin))
        - dist_n too large, dist_p too small
        - margin too small
        - the categories of positive and negative samples are not close neighbors
        - the selection of positive and negative samples is not on the same side


### Reference List
01. [Deep Learning of Binary Hash Codes for Fast Image Retrieval](http://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf)
02. [Deep Relative Distance Learning- Tell the Difference Between Similar Vehicles](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf)
03. [Deep Supervised Discrete Hashing](https://arxiv.org/abs/1705.10999)
04. [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
05. [FaceNet- A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
06. [Fast Training of Triplet-based Deep Binary Embedding Networks ](https://arxiv.org/abs/1603.02844)
07. [Hard-Aware Deeply Cascaded Embedding](https://arxiv.org/abs/1611.05720)
08. [HashNet: Deep Learning to Hash by Continuation](https://arxiv.org/abs/1702.00758)
09. [Fast Supervised Hashing with Decision Trees for High-Dimensional Data](https://arxiv.org/pdf/1404.1561.pdf)
10. [Simultaneous Feature Learning and Hash Coding with Deep Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.pdf)
11. [Learning to Hash with Binary Reconstructive Embeddings](https://papers.nips.cc/paper/3667-learning-to-hash-with-binary-reconstructive-embeddings)

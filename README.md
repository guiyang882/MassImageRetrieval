### 模型设计的指导
1. 修改采样的方案，通过每隔几轮的更新候选集合进行采样
    1.1 采样中当选择了(x_a, x_p)之后，如何确定选择的x_n是一个可以提升结果的点
2. 细化case方案，重新定制损失函数，把损失函数可视化出来
3. 设计x_a, x_p, x_n之间的矢量信息，求出夹角方向值，重新设计损失函数
4. 通过增大的batch信息，将类内误差和类间误差添加到损失函数中去


### 问题以及解决？
1. 所有的训练样本都是根据随机选择的，其中存在部分数据是很难被直接选择到的，导致10分类的分类器的分类性能下降
2. 改进样本构造的方案，使得所有的样本都可以进入分类器进行训练

### 实验结果
* Triple Loss + Classification + Cosine Random Samples
![image](https://github.com/liuguiyangnwpu/MassImageRetrieval/blob/master/experiment/showImages/triple_classifiy.png)

### TODOLIST
- [x] 使用Res50提取图像的特征
- [x] 编写孪生网络进行测试
- [x] 编写Triple Loss网络，并进行测试
- [ ] 重新设计Triple Loss网络训练样本的构造
    - [x] 添加了基于聚类中心的anchor选择和在给定半径之外的正负样本的选择
    - [x] 添加了针对训练样本中`$(x_a, x_p, x_n)$`之间的方向条件进行选择
    - [ ] 添加针对Query列表候选集进行训练样本选择的策略
- [ ] 根据TripleModel输入的数据中可以转化成PairWise的排序问题


### PaperList
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

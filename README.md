# 基于PCA的手写数字与人脸识别 
本项目是2020年春秋学期课程 *数据挖掘与机器学习* 的一次课程作业。作业要求实现PCA算法并用来处理[MNIST手写数字数据集](http://deeplearning.net/data/mnist/mnist.pkl.gz)和[AT&T人脸数据集](https://www.r-bloggers.com/wp-content/uploads/2010/09/ATTfaces.tar.gz)。首先使用PCA对数据进行压缩，然后再进行识别。识别算法在本项目中使用SVM。

## 安装

本项目使用python3作为编程语言，版本号为3.7.3。项目所需的依赖包在文件[requirements.txt](./requirements.txt)中列出，建立虚拟环境后，使用以下命令来进行安装：

`pip install -r requirements.txt`

## 使用说明

个人实现的PCA算法在文件[pca.py](./pca.py)中，实现了PCA的最基本功能，但是效果较已有的工具包中的PCA算法较差，原因暂没有详细研究。[utilities.py](./utilities.py)提供了一系列的函数，具体实现了PCA分析的各个功能，详细可以参与代码中的备注。程序入口[main.py](./main.py)提供了两个函数分别进行了对MNIST数据集和AT&T数据集的分析，可以通过注释来分别运行。


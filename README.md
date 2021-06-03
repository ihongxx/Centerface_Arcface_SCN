# Introduction

本项目主要是对人脸检测和人脸识别项目的整合，人脸检测模型centerface已经转换成onnx格式，人脸识别模型arcface也转换成onnx格式，这样做大大节省了模型调用的时间。
为了能够让模型在生产环境中一直处于预热状态，并且不需要反复的调用模型，因此采用单列模型将模型启动。

### 人脸库中添加人脸图像
直接在dataset/images下创建人名文件，并在该文件中放入含有人脸的图片

### 对人脸库中人脸特征向量提取保存
$ python create_dataset.py  # 对人脸库dataset/images下每个图的人脸进行检测并且提取人脸特征向量保存下来

### 下载已经训练好的模型到checkpoints文件夹下
链接: https://pan.baidu.com/s/1EtzHs6uU5uMWBaEyrsHPyw  
密码: c0qa

## 测试
$ python demo.py

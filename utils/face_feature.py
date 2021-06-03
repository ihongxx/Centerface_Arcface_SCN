import os
import cv2
import torch
import numpy as np

def load_image(image_path):
    """人脸库读取图像方法"""
    image = cv2.imread(image_path)
    image = cv2.resize(image,(128,128))  # (128, 128, 3)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image
def get_featuresdict(model, file):
    """对人脸库进行图片特征向量提取"""
    list_file = os.listdir(file)
    person_dict = {}
    for each in list_file:
        image = load_image(os.path.join(file, each))

        data = torch.from_numpy(image)
        # data = data.to(torch.device("cuda"))
        output = model(data)
        output = output.data.cpu().numpy()  # 将输出转为numpy

        fe_1 = output[0]
        fe_2 = output[1]
        feature = np.hstack((fe_1, fe_2))
        person_dict[each] = feature
        # print("人脸库中的特征向量：", person_dict[each])
    return person_dict

def load_image_array(image):
    """对每个人脸图片进行操作"""
    # print("images_shape_load_model_1:", image.shape)  # (128, 128, 3)

    image = np.dstack((image, np.fliplr(image)))
    # print("image.shape_1:", image.shape)  # image.shape_1: (128, 128, 6)
    image = image.transpose((2, 0, 1))
    # print("image.shape_2:", image.shape)  # image.shape_2: (6, 128, 128)
    image = image[:, np.newaxis, :, :]
    # print("image.shape_3:", image.shape)  # image.shape_3: (6, 1, 128, 128)
    image = image.astype(np.float32, copy=False)
    # print("image.shape_4:", image.shape)  # image.shape_4: (6, 1, 128, 128)
    image -= 127.5
    # print("image.shape_5:", image.shape)  # image.shape_5: (6, 1, 128, 128)
    image /= 127.5
    # image.shape: (6, 1, 128, 128)
    return image

def get_features(ort_session, face_images_list, input_name, outputs):
    """对需要检测的人脸进行特征提取"""
    """这是arcface onnx版本对测试图片进行特征提取"""
    features = []
    for i, image_array in enumerate(face_images_list):
        image = load_image_array(image_array)
        # print("image.shape:", image.shape)

        output = ort_session.run([outputs], input_feed={input_name: image})
        fe_1 = output[0][0]
        fe_2 = output[0][1]

        feature = np.hstack((fe_1, fe_2))
        features.append(feature)

        # print("对检测到的人脸特征向量：", features)

    return features

def compare_embadding(pred_emb, dataset_emb, names_list, threshold=0.65):
    """
        将待识别的人脸特征向量与人脸数据库进行比较, 这是预先已经对人脸库进行检测和识别，特征向量保存在npy文件中
        :param pred_emb: 待识别的人脸_list
        :param dataset_emb: 人脸数据库特征向量
        :param threshold： 置信阈值，小于说明是同一个人，大于说明不是同一个人
        :return:预测人名，预测分值
    """
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    # print("pred_emb:", pred_emb)
    dataset_num = len(dataset_emb)
    # print(dataset_num)
    # print("dataset_emb:", dataset_emb)
    pred_name = []
    pred_score = []

    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i], dataset_emb[j]))))
            dist_list.append(dist)
        # print("dist_list:", dist_list)
        min_value = min(dist_list)  # 检测到的人脸特征向量与人脸库向量进行对比，找到最短距离
        pred_score.append(min_value)
        # if (min_value > threshold):
        #     pred_name.append("unknow")
        # else:
        #     pred_name.append(names_list[dist_list.index(min_value)])
        pred_name.append(names_list[dist_list.index(min_value)])  # 根据检测的人脸找到与人脸库中最短的那个人脸，得到那个索引，根据索引获得名字
        # print("pred_name:", pred_name)
    return pred_name, pred_score

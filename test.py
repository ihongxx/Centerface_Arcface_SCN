import cv2
import time
import numpy as np
from centerface_singleton import centerface
from arcface_singleton import arcface
from utils import image_processing
from utils.image_processing import read_data
from utils.face_feature import compare_embadding

def load_dataset(dataset_path, filename):
    """
    用于人脸识别：对人脸库中的特征向量进行加载
    :param dataset_path:
    :param filename:
    :return:
    """
    emdeddings = np.load(dataset_path)
    names_list = read_data(filename, split=None, convertNum=False)
    return emdeddings, names_list

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

def process(test_image_path):
    image = cv2.imread(test_image_path)
    h, w = image.shape[:2]

    # 调用人脸检测
    time2 = time.time()
    dets, lms = centerface(image, h, w, threshold=0.35)
    # print("centerface1_id:", id(centerface))
    face_images_list1 = image_processing.get_bboxes_image(image, dets, resize_height=128, resize_width=128)
    time3 = time.time()
    print("人脸检测所需时间：", time3-time2)

    # 遍历人脸,并且对检测的人脸特征提取
    time4 = time.time()
    print("arcface_id:", id(arcface))
    input_name = arcface.get_inputs()[0].name
    outputs = arcface.get_outputs()[0].name
    features = []
    for i, image_array in enumerate(face_images_list1):
        image = load_image_array(image_array)

        output = arcface.run([outputs], input_feed={input_name:image})
        fe_1 = output[0][0]
        fe_2 = output[0][1]

        feature = np.hstack((fe_1, fe_2))
        features.append(feature)
    time5 = time.time()
    print("遍历人脸，并提取人脸特征时间：", time5 - time4)

    # 得到人脸库的所有的特征
    dataset_path = 'dataset/emb/faceEmbedding.npy'
    filename = 'dataset/emb/name.txt'
    time6 = time.time()
    dataset_emb, names_list = load_dataset(dataset_path, filename)  # 预先已经对人脸库进行检测与识别，直接加载人脸库的特征向量
    time7 = time.time()
    print("加载人脸库特征时间：", time7-time6)

    # 对检测到的人脸特征与人脸库特征进行匹配
    time8 = time.time()
    pred_name, pred_score = compare_embadding(features, dataset_emb, names_list)
    time9 = time.time()
    print("对比检测到的人脸特征与人脸库的特征所需时间:", time9-time8)
    print("pred_name:", pred_name)

if __name__ == '__main__':
    time1 = time.time()

    test_image_path1 = 'dataset/test_images/1.jpg'
    # test_image_path2 = 'dataset/test_images/2.jpg'

    time20 = time.time()
    process(test_image_path1)
    time21 = time.time()
    print("线程第一次处理时间：", time21-time20)

    # time22 = time.time()
    # process(test_image_path2)
    # time23 = time.time()
    # print("线程第二次处理时间：", time23 - time22)

    time10 = time.time()
    print("处理总时间：",time10-time1)
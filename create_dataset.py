"""
    对人脸库的的图片进行检测与识别，并将特征向量保存下来
"""
import torch
import cv2
import numpy as np
from utils.file_processing import gen_files_labels, write_list_data
from config import Config
from centerface import CenterFace
from resnet import resnet_face18
from torch.nn import DataParallel
from utils import image_processing
from utils.face_feature import get_features

def get_face_embedding(files_list, names_list):
    """
    获得embedding数据
    :param files_list: 图像列表
    :param names_list: 与files_list--的名称列表
    :return:
    """
    # 初始化arcface模型
    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path))
    # model.to(torch.device("cuda"))
    model.eval()

    embeddings = []  # 用于保存人脸特征数据库
    label_list = []  # 保存人脸label的名称，与embeddings--对应
    for image_path, name in zip(files_list, names_list):
        print("processing image: {}".format(image_path), "image_name:", name)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # 初始化centerface人脸检测模型
        landmarks = True  # True表示是否需要关键点
        centerface = CenterFace(landmarks=landmarks)
        dets, lms = centerface(image, h, w, threshold=0.35)   # 通过centerface得到人脸框和关键点信息

        if dets == [] or lms == []:
            print("-----no face")
            continue
        if len(dets) >= 2:
            print("-----image have {} faces".format(len(dets)))
            continue
        # 得到截取的矩形人脸图片列表,并且resize为128×128
        face_images_list = image_processing.get_bboxes_image(image, dets, resize_height=128, resize_width=128)
        # print("获得人脸列表：", face_images_list)
        # 获得人脸特征
        pred_emb = get_features(model, face_images_list)
        embeddings.append(pred_emb)
        label_list.append(name)
    return embeddings, label_list

def create_face_embedding(dataset_path, out_emb_path, out_filename):
    """
    :param dataset_path: 人脸数据库路径，每一类单独一个文件夹
    :param out_emb_path:输出embeddings的路径
    :param out_filename:输出与embeddings--对应的标签a
    :return:
    """
    files_list, names_list = gen_files_labels(dataset_path, postfix=['*.jpg'])
    print("file_list:", files_list, "name_list:", names_list)
    embeddings, label_list = get_face_embedding(files_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))

    embeddings = np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    write_list_data(out_filename, label_list, mode='w')


if __name__ == '__main__':
    dataset_path = 'dataset/images/'  # 人脸库的路径
    out_emb_path = 'dataset/emb/faceEmbedding.npy'  # 人脸库特征向量的向量保存文件
    out_filename = 'dataset/emb/name.txt'  # 人脸库对应的人名保存txt文件
    create_face_embedding(dataset_path, out_emb_path, out_filename)
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

def get_bboxes_image(image, bboxes_list, resize_height=None, resize_width=None):
    rects_list = bboxes2rects(bboxes_list)
    rect_images = get_rects_image(image, rects_list, resize_height, resize_width)
    return rect_images

def bboxes2rects(bboxes_list):
    """
    将bboxes=[x1,y1,x2,y2] 转为rect=[x1,y1,w,h]
    :param bboxes_list: [x1,y1,x2,y2]的人脸列表
    :return: [x1,y1,w,h]的人脸列表
    """
    rect_list = []
    for bbox in bboxes_list:
        x1, y1, x2, y2 = bbox[:4]
        rect = [x1, y1, (x2-x1), (y2- y1)]
        rect_list.append(rect)
    return rect_list

def get_rects_image(image, rects_list, resize_height=None, resize_width=None):
    """
    得到人脸图像列表
    :param image: 原始图片
    :param rects_list: [x1,y1,w,h]的人脸列表
    :param resize_height:
    :param resize_width:
    :return:
    """
    rect_images = []
    for rect in rects_list:
        roi = get_rect_image(image, rect)
        roi = cv2.resize(roi, dsize=(resize_width, resize_height))
        rect_images.append(roi)
    return rect_images

def get_rect_image(image, rect):
    """得到一个人脸图像"""
    shape = image.shape
    height = shape[0]
    width = shape[1]
    image_rect = (0, 0, width, height)
    rect = get_rect_intersection(rect, image_rect)  # 得到人脸框框
    x, y, w, h = rect
    cut_img = image[int(y):int((y + h)), int(x):int((x + w))]
    return cut_img

def get_rect_intersection(rec1, rec2):
    """计算两个rect的交集"""
    cx1, cy1, cx2, cy2 = rects2bboxes([rec1])[0]
    gx1, gy1, gx2, gy2 = rects2bboxes([rec2])[0]
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return (x1, y1, w, h)

def rects2bboxes(rects_list):
    """将rect=[x1,y1,w,h]转为bboxes=[x1,y1,x2,y2]"""
    bboxes_list = []
    for rect in rects_list:
        # print("erect:",rect)
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h
        b = (x1, y1, x2, y2)
        bboxes_list.append(b)
    return bboxes_list

def face_analysis_image_process(image_array):
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # img = Image.open(image_path).convert('RGB')
    # img = np.array(image_path)
    img = preprocess_transform(image_array)
    # print("img:", img)
    return img

def cv_show_image(title, image, type='rgb'):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :param type:'rgb' or 'bgr'
    :return:
    '''
    channels=image.shape[-1]
    if channels==3 and type=='rgb':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_image_bboxes_text(title, rgb_image, boxes, boxes_name):
    '''
    :param boxes_name:
    :param bgr_image: bgr image
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    print("boxes:", boxes)
    for name ,box in zip(boxes_name,boxes):
        box=[int(b) for b in box]
        cv2.rectangle(bgr_image, (box[0],box[1]),(box[2],box[3]), (0, 255, 0), 2, 8, 0)
        cv2.putText(bgr_image,name, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv_show_image(title, rgb_image)

def is_int(str):
    # 判断是否为整数
    try:
        x = int(str)
        return isinstance(x, int)
    except ValueError:
        return False

def is_float(str):
    # 判断是否为整数和小数
    try:
        x = float(str)
        return isinstance(x, float)
    except ValueError:
        return False

def read_data(filename, split=" ", convertNum=True):
    """
    读取txt数据函数
    :param filename: 文件名
    :param split: 分割符
    :param convertNum: 是否将list中的string转为int/float类型的数字
    :return:
    """
    with open(filename, mode='r', encoding='utf-8') as f:
        content_list = f.readlines()
        if split is None:
            content_list = [content.rstrip() for content in content_list]  # rstrip() 删除 string 字符串末尾的指定字符（默认为空格）.
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]  # 指定分隔符对字符串进行切片
        if convertNum:
            for i, line in enumerate(content_list):
                line_data = []
                for l in line:
                    if is_int(l):  # isdigit() 方法检测字符串是否只由数字组成,只能判断整数
                        line_data.append(int(l))
                    elif is_float(l):  # 判断是否为小数
                        line_data.append(float(l))
                    else:
                        line_data.append(l)
                content_list[i] = line_data
    return content_list

import threading
import onnxruntime as ort
from centerface import CenterFace

def synchronized(func):
    func.__lock__ = threading.Lock()  # 线程锁的实现

    def synced_func(*args, **kwargs):
        with func.__lock__:  # with自动打开释放线程锁
            return func(*args, **kwargs)

    return synced_func

class centerface_singleton(object):

    _instance = None

    def __init__(self, *args, **kwargs):
        pass

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # 如果类实例化为None

            # 初始化人脸检测模型
            landmarks = True
            centerface = CenterFace(landmarks)

            cls._instance = centerface

        return cls._instance

centerface = centerface_singleton()
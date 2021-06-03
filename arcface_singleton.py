import threading
import onnxruntime as ort

def synchronized(func):
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return synced_func

class arcface_singleton(object):

    _instance = None

    def __init__(self, *args, **kwargs):
        pass

    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:

            # 初始化arcface人脸检测模型
            onnx_file = 'checkpoints/arcface_dynamic.onnx'
            ort_session = ort.InferenceSession(onnx_file)

            cls._instance = ort_session

        return cls._instance

arcface = arcface_singleton()
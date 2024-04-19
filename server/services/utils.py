import numpy as np
import cv2

def protomat2numpy(proto_mat):
    shape = proto_mat.shape
    data = np.asarray(proto_mat.data, dtype=float).reshape(shape)
    return data
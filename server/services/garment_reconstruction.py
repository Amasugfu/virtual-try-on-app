from ...public.protobuf.python import (
    requests_pb2_grpc,
    requests_pb2,
)
from .. import utils
import pickle

class XCloth:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self) -> None:
        pass

    def generate_model_3d(self, img):
        pass

class GarmentReconstruction(requests_pb2_grpc.GarmentReconstructionServicer):

    def getModel3D(self, request, context):
        model3d, texture = XCloth().generate_model_3d(
            request.data
        )
        return utils.convert_model3d_grpc(model3d, texture)
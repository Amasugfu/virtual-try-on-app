import numpy as np

import grpc
from concurrent import futures
from ..proto import (
    requests_pb2,
    requests_pb2_grpc
)

import cv2
from romp import ROMP, romp_settings

from .utils import protomat2numpy
from models.xcloth.production import Pipeline

class GarmentReconstructionServicer(requests_pb2_grpc.GarmentReconstructionServicer):
    CHUNK_SIZE = 1024*1024 # 1 MB
    
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline
    
    def reconstruct(self, request, context):
        img = np.flip(protomat2numpy(request.garment_img), axis=0)
        pose = protomat2numpy(request.pose)
        
        gltf = self._pipeline(img, pose)
        
        for i in range(0, len(gltf), self.CHUNK_SIZE):
            yield requests_pb2.Model3D(
                size = len(gltf),
                data = gltf[i:i + self.CHUNK_SIZE]   
            )
    
    
class PoseDetectionServicer(requests_pb2_grpc.PoseDetectionServicer):
    def __init__(self, romp=ROMP(romp_settings(["--mode=webcam"]))) -> None:
        super().__init__()
        self._romp = romp
        
    def getPose(self, request, context):
        img = np.frombuffer(request.buffer[0], dtype=np.uint8).reshape(720, 1280, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        out = self._romp(np.transpose(img, (1, 0, 2))[::-1, ::-1, :])
        
        return requests_pb2.FloatMat(
            num_dim = 3,
            shape = [24, 4, 4],
            data = np.zeros(384)
        )

        
def serve(port=50000, pipeline: Pipeline = Pipeline()):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    requests_pb2_grpc.add_GarmentReconstructionServicer_to_server(
        GarmentReconstructionServicer(pipeline=pipeline),
        server
    )
    requests_pb2_grpc.add_PoseDetectionServicer_to_server(
        PoseDetectionServicer(),
        server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
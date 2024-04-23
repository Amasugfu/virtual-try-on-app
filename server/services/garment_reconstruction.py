import numpy as np

import grpc
from concurrent import futures
from ..proto import (
    requests_pb2,
    requests_pb2_grpc
)

import cv2
from romp import ROMP, romp_settings

from dataclasses import dataclass
from typing import Any

from .utils import protomat2numpy
from models.xcloth.production import Pipeline
from models.rigging.utils import paint_mesh_to_glb


@dataclass
class UserCache:
    client_id: str
    input_pose: np.ndarray
    output_mesh: Any


class GarmentReconstructionServicer(requests_pb2_grpc.GarmentReconstructionServicer):
    CHUNK_SIZE = 1024*1024 # 1 MB
    
    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._client_cache = {}
        
    def transfer_weights(self, client_id):
        # #############################################################
        # ### debug block
        # with open("debug_results/result_rigged.glb", "rb") as f:
        #     return f.read()
        # #############################################################
        
        cache = self._client_cache[client_id]
        return paint_mesh_to_glb(cache.output_mesh, cache.input_pose)
    
    def reconstruct(self, request, context):
        client_id = context.peer()
        
        pose = protomat2numpy(request.pose)
        
        if pose.size == 1:
            # confirm cache               
            if pose.item() > 0:
                gltf = self.transfer_weights(client_id)
            elif client_id in self._client_cache:
                del self._client_cache[client_id]
        else:
            # reconstruct
            img = np.flip(protomat2numpy(request.garment_img), axis=0)
            smpl_pose = self.to_smpl_pose(pose)
            gltf, mesh = self._pipeline(img, pose, smpl_pose)
            self._client_cache[client_id] = UserCache(client_id=client_id, input_pose=smpl_pose, output_mesh=mesh)
        
        # yield stream
        size = len(gltf)
        i = 0
        while size > 0:
            yield requests_pb2.Model3D(
                size = len(gltf),
                data = gltf[i:i + self.CHUNK_SIZE]   
            )
            i += self.CHUNK_SIZE
            size -= self.CHUNK_SIZE
            
    @staticmethod
    def to_smpl_pose(pose):
        smpl_pose = np.zeros((24, 3))
        smpl_pose[16, 2] = pose[-4]
        smpl_pose[17, 2] = pose[-3]
        smpl_pose[1, 2] = pose[-2]
        smpl_pose[2, 2] = pose[-1]
        smpl_pose = np.deg2rad(smpl_pose) * -1
        return smpl_pose
    
class PoseDetectionServicer(requests_pb2_grpc.PoseDetectionServicer):
    def __init__(self, romp=ROMP(romp_settings(["--mode=webcam", "--show"]))) -> None:
        super().__init__()
        self._romp = romp
        
    def getPose(self, request, context):
        img = np.frombuffer(request.buffer[0], dtype=np.uint8).reshape(720, 1280, 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        out = self._romp(np.transpose(img, (1, 0, 2))[::-1, ::-1, :])
        pose = np.hstack([out["global_orient"], out["body_pose"]]).squeeze()
        pose = np.rad2deg(pose)
        trans = (out["cam_trans"] + out["joints"][:, 0]).squeeze()

        return requests_pb2.FloatMat(
            num_dim = 1,
            shape = [75],
            data = np.hstack([trans, pose])
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
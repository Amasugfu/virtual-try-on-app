from ..public.protobuf.python import (
    requests_pb2_grpc,
    requests_pb2,
)

def split_bytes(b, n):
    return [b[i:i+n] for i in range(0, len(b), n)]


def convert_model3d_grpc(model, texture):
    data = split_bytes(model)
    model3d = requests_pb2.Model3D(size=len(data), texture=texture)
    model3d.data.extend(data)
    return model3d


def convert_image2d_grpc(img):
    return
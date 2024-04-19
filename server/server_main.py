from .services import garment_reconstruction

from models.xcloth.production import Pipeline, XCloth

class ServerEntry:
    def __init__(self) -> None:
        pass
    

def main(checkpoint_path: str | None = None, port=50000):
    if checkpoint_path is not None:
        model = XCloth()
        model.load(checkpoint_path)
        pipeline = Pipeline(model=model)
    else:
        pipeline = Pipeline()
    
    garment_reconstruction.serve(
        port=port,
        pipeline=pipeline
    )
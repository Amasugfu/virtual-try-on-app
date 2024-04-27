from .services import garment_reconstruction

from models.xcloth.production import Pipeline, XCloth

def main(checkpoint_path: str | None = None, port=50000):
    """main server entry point

    Parameters
    ----------
    checkpoint_path : str | None, optional
        xcloth model paramters, by default None
    port : int, optional
        server port number, by default 50000
    """
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
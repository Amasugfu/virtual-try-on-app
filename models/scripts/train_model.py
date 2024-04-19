import argparse
# import dependency
import os, sys
from os.path import abspath, dirname, join
m_path = dirname(abspath(join(__file__, "..")))
if m_path not in sys.path:
    sys.path.append(m_path)

import logging
from logging import StreamHandler, FileHandler, Formatter

def setup_logger(args):
    logger = logging.getLogger("xcloth")
    logger.setLevel(logging.DEBUG)

    fmt = "[%(asctime)s] %(message)s"
    formatter = Formatter(fmt=fmt)
    if args.verbose: 
        handler = StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if args.log_file is not None: 
        handler = FileHandler(args.log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def main(args):
    logger = setup_logger(args)

    logger.info("importing dependencies...")

    from xcloth.train.data import MeshDataSet
    from xcloth.production import XCloth
    from xcloth.train.trainer import train_model
    import torch
    torch.manual_seed(0)
    
    logger.info("done")

    logger.info("loading dataset...")
    data_path = args.path
    mask = None if args.mask is None else set(args.mask)
    dataset = MeshDataSet(root_dir=data_path, mask=mask, excld=args.exclude, depth_offset=float(args.depth))
    logger.info(dataset.stats)
    if args.split is not None:
        frac = float(args.split)
        dataset, test_dataset = torch.utils.data.random_split(dataset, [frac, 1 - frac])

    logger.info("done")

    logger.info("training model...")

    model = XCloth().cuda()
    n_epoch = int(args.n_epoch)
    lr = float(args.lr)

    n = 1
    if args.recover:
        n, _ = model.load(args.checkpoint)
        lr *= 0.95**n

    
    if n < n_epoch:
        train_model(
            model,
            dataset,
            batch_size=int(args.batch_size),
            start_epoch=n + 1,
            n_epoch=n_epoch,
            lr=lr,
            reduction="mean",
            weight=[1., 0.1, 1., 0.05, 1, 0.5],
            separate_bg=True,
            params_path=args.checkpoint,
            plot_path=args.plot_path,
            logger=logger,
        )
       
    logger.info("done")

    if args.test is not None:
        logger.info("testing model...")
        from xcloth.components.utils import GarmentModel3D
        with torch.no_grad():
            for idx, (x, _, _, _) in enumerate(test_dataset):
                x = x.cuda()
                r = model(x[:3], x[3:])
                mesh = GarmentModel3D.from_tensor_dict(r, 1)
                p = f"{args.test}/{idx}"
                mesh[0].backproject(thres=0.25, depth_offset=float(args.depth), path=p)
            logger.info(f"saving test data {idx} into {p}")

        logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-d", "--depth")
    parser.add_argument("-n", "--n_epoch", default=20)
    parser.add_argument("-b", "--batch_size", default=4)
    parser.add_argument("-l", "--lr", default=0.0005)
    parser.add_argument("-m", "--mask", nargs="+")
    parser.add_argument("-e", "--exclude", nargs="?", default=False, const=True)
    parser.add_argument("-v", "--verbose", nargs="?", default=False, const=True)
    parser.add_argument("-r", "--recover", nargs="?", default=False, const=True)
    parser.add_argument("-t", "--test")
    parser.add_argument("-s", "--split")
    parser.add_argument("--log_file")
    parser.add_argument("--plot_path")
    args = parser.parse_args()

    main(args)
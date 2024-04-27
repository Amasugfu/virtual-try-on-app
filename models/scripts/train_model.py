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
    from xcloth.train import train_model
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

        # load grouping info and split according to groups
        if os.path.exists(group_info := os.path.join(data_path, "groups.json")):
            import json, random, math
            random.seed(0)

            with open(group_info, "r") as f:
                group_info = json.load(f)
            train_ids = []
            test_ids = []
            for g in group_info:
                pool = group_info[g].split()
                selected = random.sample(pool, k=math.ceil(len(pool) * frac))
                train_ids.extend(selected)
                test_ids.extend([i for i in pool if i not in set(selected)])

            train_dataset = torch.utils.data.Subset(dataset, dataset.find_indices_of_ids(train_ids))
            test_dataset = torch.utils.data.Subset(dataset, dataset.find_indices_of_ids(test_ids))

            logger.info(f"train set ids: {train_ids}")        
            logger.info(f"test set ids: {test_ids}")
        else:
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [frac, 1 - frac])

    logger.info(f"train set size: {len(train_ids)}({len(train_dataset)})")
    logger.info(f"test set size: {len(test_ids)}({len(test_dataset)})")

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
            train_dataset,
            batch_size=int(args.batch_size),
            start_epoch=n,
            n_epoch=n_epoch,
            lr=lr,
            reduction="mean",
            weight=[1., 0.1, 1., 0.5, 1, 0.5],
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
            for idx, (name, x, _, _, _) in enumerate(test_dataset):
                x = x.cuda()
                r = model(x[:3], x[3:])
                mesh = GarmentModel3D.from_tensor_dict(r, 1)
                p = f"{args.test}/{idx}_{name}"
                mesh[0].reconstruct(thres=0.25, depth_offset=float(args.depth), path=p)
                logger.info(f"saving test data {name} ({idx}) into {p}")

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
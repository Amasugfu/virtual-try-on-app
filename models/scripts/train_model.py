import argparse
# import dependency
import os, sys
from os.path import abspath, dirname, join
m_path = dirname(abspath(join(__file__, "..")))
if m_path not in sys.path:
    sys.path.append(m_path)


def main(args):
    if args.verbose: print("[INFO] importing dependencies")
    from xcloth.train.data import MeshDataSet
    from xcloth.production import XCloth
    from xcloth.train import train_model
    if args.verbose: print("[INFO] Done", end="\n\n")

    if args.verbose: print("[INFO] loading dataset")
    data_path = args.path
    mask = None if args.mask is None else set(args.mask)
    dataset = MeshDataSet(root_dir=data_path, mask=mask, excld=args.exclude)
    dataset.make_Xy(depth_offset=.5)

    if args.verbose: 
        print(dataset.stats)
        print("[INFO] Done", end="\n\n")
        print("[INFO] training model")

    model = XCloth().cuda()
    n_epoch = int(args.n_epoch)
    lr = float(args.lr),

    if args.recover:
        n = model.load(args.checkpoint)
        n_epoch -= n
        lr *= 0.95**(n)

    train_model(
        model,
        dataset,
        batch_size=int(args.batch_size),
        n_epoch=n_epoch,
        lr=lr,
        reduction="mean",
        weight=[1., 0.1, 1., 0.05, 1, 0.5],
        separate_bg=True,
        params_path=args.checkpoint,
        verbose=args.verbose,
        plot_path=args.plot_path
    )
       
    if args.verbose: print("[INFO] Done", end="\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-n", "--n_epoch", default=20)
    parser.add_argument("-b", "--batch_size", default=4)
    parser.add_argument("-l", "--lr", default=0.0005)
    parser.add_argument("-m", "--mask", nargs="+")
    parser.add_argument("-e", "--exclude", nargs="?", default=False, const=True)
    parser.add_argument("-v", "--verbose", nargs="?", default=False, const=True)
    parser.add_argument("-r", "--recover", nargs="?", default=False, const=True)
    parser.add_argument("--plot_path")
    args = parser.parse_args()

    main(args)
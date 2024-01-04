import argparse
# import dependency
import os, sys
from os.path import abspath, dirname, join
m_path = dirname(abspath(join(__file__, "..")))
if m_path not in sys.path:
    sys.path.append(m_path)


from xcloth.train.data import DataLoader


def make_truth(in_dir, out_dir):
    print(f"-- started loading from <{in_dir}> --")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    loader = DataLoader()
    loader.load_n_process(in_dir, out_dir, verbose=True, log_file="processed.log")
    print(f"-- finished --")


def make_input(in_dir, out_dir):
    pass


def main(args):
    for in_dir, out_dir, target in zip(args.input_folder, args.output_folder, args.target):
        locals()[f"make_{target}"](in_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", nargs="+", default=["input", "truth"], required=True)
    parser.add_argument("-i", "--input-folder", nargs="+", required=True)
    parser.add_argument("-o", "--output-folder", nargs="+", required=True)
    args = parser.parse_args()

    main(args)
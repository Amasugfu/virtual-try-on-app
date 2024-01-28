import argparse
# import dependency
import os, sys
from os.path import abspath, dirname, join
m_path = dirname(abspath(join(__file__, "..")))
if m_path not in sys.path:
    sys.path.append(m_path)


def main(args):
    for in_dir, out_dir, target in zip(args.input_folder, args.output_folder, args.target):
        print(f"-- started loading [{target}] from [{in_dir}] --")

        from xcloth.train.data import DataProccessor

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        loader = DataProccessor()
        loader.process_data(in_dir, out_dir, whitelist=args.whitelist_path, target=target, no_replace=not args.force_replace, verbose=args.verbose, log_file=args.log_file, smpl_path=args.smpl_path)
        print(f"-- finished --")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", nargs="+", default=["input", "truth"], required=True)
    parser.add_argument("-i", "--input-folder", nargs="+", required=True)
    parser.add_argument("-o", "--output-folder", nargs="+", required=True)
    parser.add_argument("-v", "--verbose", nargs="?", default=False, const=True)
    parser.add_argument("-l", "--log-file")
    parser.add_argument("--force-replace", nargs="?", default=False, const=True)
    parser.add_argument("--n-task", default=4)
    parser.add_argument("--smpl-path")
    parser.add_argument("--whitelist-path")
    args = parser.parse_args()

    main(args)
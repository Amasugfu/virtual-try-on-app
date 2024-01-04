import argparse
# import dependency
import os, sys
from os.path import abspath, dirname, join
m_path = dirname(abspath(join(__file__, "..")))
if m_path not in sys.path:
    sys.path.append(m_path)


from xcloth.train.data import DataLoader


def main(in_dir, out_dir):
    print(f"-- started loading from <{in_dir}> --")
    loader = DataLoader()
    loader.load_n_process(in_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    loader.save(os.path.join(out_dir, "registered_data.pickle"))
    print(f"-- finished. saving to <{out_dir}> --")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-folder", required=True)
    parser.add_argument("-o", "--output-folder", required=True)
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
from .server_main import main

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--port", default=50000, type=int)
    parser.add_argument("-c", "--chkpt", default=None, type=str)
    
    args = parser.parse_args()
    
    main(checkpoint_path=args.chkpt, port=args.port)
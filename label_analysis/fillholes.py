import argparse
from pathlib import Path

from utilz.helpers import multiprocess_multiarg
from fillholes_worker import fill_holes_multiclass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", nargs="?", type=Path)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    fldr = cli_args.input_folder
    fldr_out = cli_args.output_folder or fldr
    fldr_out.mkdir(parents=True, exist_ok=True)
    args = [[fn, fldr_out/fn.name] for fn in fldr.glob("*")]
    multiprocess_multiarg(fill_holes_multiclass,args,num_processes=8)


if __name__=='__main__':
    main()

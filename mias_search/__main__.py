"""
This is the command-line interface for the MIaS Search package.
"""

from argparse import ArgumentParser
import gzip
import logging
from logging import getLogger
from pathlib import Path
import pickle
from sys import stdout
from urllib.parse import urlparse


LOG_PATH = Path("__main__.log")
LOG_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
LOGGER = getLogger(__name__)
ROOT_LOGGER = getLogger()


def main():
    """ Main entry point of the app """
    ROOT_LOGGER.setLevel(logging.DEBUG)

    file_handler = logging.StreamHandler(LOG_PATH.open("wt"))
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    ROOT_LOGGER.addHandler(file_handler)

    terminal_handler = logging.StreamHandler(stdout)
    terminal_handler.setFormatter(formatter)
    terminal_handler.setLevel(logging.INFO)
    ROOT_LOGGER.addHandler(terminal_handler)

    LOGGER.debug("Parsing command-line arguments")
    parser = ArgumentParser(
        description="""
            Use topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format to query
            the WebMIaS interface of the MIaS Math Information Retrieval system and to retrieve
            result document lists.
        """)
    parser.add_argument(
        "--dataset", required=True, type=Path, help="""
            A path to a directory containing a dataset in the NTCIR-11 Math-2, and NTCIR-12 MathIR
            XHTML5 format.
        """)
    parser.add_argument(
        "--topics", required=True, type=Path, help="""
            A path to a file containing topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12
            MathIR format.
        """)
    parser.add_argument(
        "--judgements", required=True, type=Path, help="""
            A path to a file containing relevance judgements for our topics.
        """)
    parser.add_argument(
        "--positions", type=Path, required=True, help="""
            The path to the file, where the estimated positions of all paragraph identifiers from
            our dataset were stored by the NTCIR Math Density Estimator package.
        """)
    parser.add_argument(
        "--estimates", type=Path, required=True, help="""
            The path to the file, where the density, and probability estimates for our dataset were
            stored by the NTCIR Math Density Estimator package.
        """)
    parser.add_argument(
        "--webmias-url", type=urlparse, required=True, help="""
            The URL at which the WebMIaS Java Servlet has been deployed.
        """)
    parser.add_argument(
        "--output-file-basename", type=Path, required=True, help="""
            The basename of the output files in the TSV (Tab Separated Value) format. Several output
            files will be produced for varying math representation, and reranking strategies. Each
            line will contain a topic ID, a reserved value, a paragraph identifier, the rank of the
            result, a reserved value, and a reserved value. This file format is recognized by the
            MIREval tool.
        """)
    args = parser.parse_args()

    LOGGER.debug("Performing sanity checks on the command-line arguments")
    assert args.dataset.exists() or args.dataset.is_dir(), \
        "Dataset %s does not exist" % args.dataset
    assert args.topics.exists() or args.topics.is_file(), \
        "The file %s with topics does not exist" % args.topics
    assert args.judgements.exists() or args.judgements.is_file(), \
        "The file %s with judgements does not exist" % args.judgements
    assert args.positions.exists() or args.positions.is_file(), \
        "The file %s with positions does not exist" % args.positions
    assert args.estimates.exists() or args.estimates.is_file(), \
        "The file %s with estimates does not exist" % args.estimates
    assert args.output_file_basename.parents[0].exists() \
        and args.output_file_basename.parents[0].is_dir(), \
        "Directory %s, where the output TSV files are to be stored, does not exist" % \
        args.output_file_basename.parents[0]

    LOGGER.info("Unpickling %s", args.positions.name)
    with gzip.open(args.positions.open("rb"), "rb") as f:
        positions = pickle.load(f)[args.dataset]

    LOGGER.info("Unpickling %s", args.estimates.name)
    with gzip.open(args.estimates.open("rb"), "rb") as f:
        estimates = pickle.load(f)[-1]

    # TODO


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

"""
This is the command-line interface for the NTCIR MIaS Search package.
"""

from argparse import ArgumentParser
import logging
from logging import getLogger
from pathlib import Path
from sys import stdout
from urllib.parse import urlparse

from numpy import mean

from .abstract import WeightedScoreAggregationStrategy
from .facade import get_topics, get_webmias, query_webmias, rerank_and_merge_results
from .util import get_judgements, get_positions, get_estimates, log_sequence
from .view import plot_evaluation_results


LOG_PATH = Path("__main__.log")
LOG_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
LOGGER = getLogger(__name__)
NUM_ESTIMATES = 5  # The number of estimates produced by the NTCIR Math Density Estimator package
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
            XHTML5 format. The directory does not need to exist, since the path is only required for
            extracting data from the file with estimated positions of paragraph identifiers.
        """)
    parser.add_argument(
        "--topics", required=True, type=Path, help="""
            A path to a file containing topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12
            MathIR format.
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
        "--judgements", type=Path, required=True, help="""
            The path to the file containing relevance judgements for our dataset.
        """)
    parser.add_argument(
        "--webmias-url", type=urlparse, required=True, help="""
            The URL at which a WebMIaS Java Servlet has been deployed.
        """)
    parser.add_argument(
        "--webmias-index-number", type=int, default=0, help="""
            The numeric identifier of the WebMIaS index that corresponds to the dataset. Defaults to
            %(default)d.
        """)
    parser.add_argument(
        "--num-workers-querying", type=int, default=1, help="""
            The number of processes that will send queries to WebMIaS. Defaults to %(default)d. Note
            that querying, reranking, and merging takes place simmultaneously.
        """)
    parser.add_argument(
        "--num-workers-merging", type=int, default=9, help="""
            The number of processes that will rerank results. Defaults to %(default)d. Note that
            querying, reranking, and merging takes place simmultaneously.
        """)
    parser.add_argument(
        "--output-directory", type=Path, default=None, help="""
            The path to the directory, where the output files will be stored. Defaults to
            %(default)s.
        """)
    parser.add_argument(
        "--plots", type=Path, nargs='+', help="""
            The path to the files, where the evaluation results will plotted.
        """)
    args = parser.parse_args()

    LOGGER.debug("Performing sanity checks on the command-line arguments")
    assert args.topics.exists() or args.topics.is_file(), \
        "The file %s with topics does not exist" % args.topics
    assert args.positions.exists() or args.positions.is_file(), \
        "The file %s with positions does not exist" % args.positions
    assert args.estimates.exists() or args.estimates.is_file(), \
        "The file %s with estimates does not exist" % args.estimates
    assert args.judgements.exists() or args.judgements.is_file(), \
        "The file %s with relevance judgements does not exist" % args.judgements
    assert args.output_directory is None or args.output_directory.exists() and \
        args.output_directory.is_dir(), \
        "Directory %s, where the output files are to be stored, does not exist" % \
        args.output_directory
    assert args.webmias_index_number >= 0
    assert args.num_workers_querying > 0
    assert args.num_workers_merging > 0
    for plot in args.plots:
        assert plot.parents[0].exists() and plot.parents[0].is_dir(), \
            "Directory %s, where a plot is to be stored, does not exist" % \
            plot.parents[0]
        if plot.exists():
            LOGGER.warning("%s exists", plot)

    LOGGER.info("Reading relevance judgements from %s", args.judgements.name)
    with args.judgements.open("rt") as f:
        judgements = get_judgements(f)
        assert judgements, "No judgements were read"
    LOGGER.info(
        "%d judged topics and %d total judgements in %s", len(judgements),
        sum(len(topic_judgements) for topic_judgements in judgements.values()),
        args.judgements.name)

    LOGGER.info("Reading topics from %s", args.topics.name)
    topics = get_topics(args.topics, judgements)
    assert len(topics) >= 2
    LOGGER.info(
        "%d topics (%s, %s, ...) contain %d formulae, and %d keywords", len(topics), topics[0],
        topics[1], sum(len(topic.formulae) for topic in topics),
        sum(len(topic.keywords) for topic in topics))

    LOGGER.info(
        "Establishing connection with a WebMIaS Java Servlet at %s", args.webmias_url.geturl())
    webmias = get_webmias(args.webmias_url, args.webmias_index_number)

    LOGGER.info("Reading paragraph position estimates from %s", args.positions.name)
    with args.positions.open("rb") as f:
        positions_all = get_positions(f)
    assert args.dataset in positions_all
    positions = positions_all[args.dataset]
    assert positions
    identifiers = positions.keys()
    LOGGER.info("%d total paragraph identifiers in %s", len(identifiers), args.positions.name)

    LOGGER.info("Reading density, and probability estimates from %s", args.estimates.name)
    with args.estimates.open("rb") as f:
        estimates_all = get_estimates(f)
    assert len(estimates_all) == NUM_ESTIMATES
    estimates = estimates_all[-1]
    assert(len(estimates))

    LOGGER.info("Querying %s, reranking and merging results", webmias)
    results = query_webmias(
        topics, webmias, positions, estimates, args.output_directory, args.num_workers_querying)
    final_results = list(rerank_and_merge_results(
        results, identifiers, args.output_directory, args.num_workers_merging))

    math_formats = dict()
    evaluation_results = []
    for aggregation, math_format, result_lists in final_results:
        assert len(result_lists) == len(topics)
        mean_score = mean([results.evaluate() for results in result_lists])
        evaluation_results.append((aggregation, math_format, mean_score))
        if math_format not in math_formats:
            math_formats[math_format] = []
        if isinstance(aggregation, WeightedScoreAggregationStrategy):
            math_formats[math_format].append((aggregation, mean_score))
    LOGGER.info("Evaluation results:")
    lines = []
    for aggregation, math_format, score in sorted(
            evaluation_results, key=lambda x: x[2], reverse=True):
        line = "%s, %s: %0.4f" % (aggregation.identifier, math_format.identifier, score)
        lines.append(line)
    log_sequence(lines)

    if args.plots:
        figure = plot_evaluation_results(sorted(math_formats.items()))
        for plot_path in args.plots:
            LOGGER.info("Plotting %s", plot_path.name)
            figure.savefig(str(plot_path))


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

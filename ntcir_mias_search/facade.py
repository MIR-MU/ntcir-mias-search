"""
This module contains facade functions for the main command-line interface.
"""

from collections import deque, KeysView
from itertools import cycle
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from .eval import ResultList
from .query import MIaSResult, ArtificialResult, ExecutedQuery, ExecutedProcessedQuery
from .topic import Topic, Formula
from .util import write_tsv
from .webmias import WebMIaSIndex, TARGET_NUMBER_OF_RESULTS


LOGGER = getLogger(__name__)
PATH_FINAL_RESULT = "final_%s.%s.tsv"


def _query_webmias_helper(args):
    topic, math_format, webmias, output_directory = args
    executed_queries = []
    for query in topic.get_queries(math_format):
        if output_directory:
            query.save(output_directory)
        executed_queries.append(ExecutedQuery.from_webmias(query, webmias))
    return (math_format, topic, executed_queries)


def query_webmias(topics, webmias, positions, estimates, output_directory=None, num_workers=1):
    """
    Produces queries from topics, queries a WebMIaS index, and returns the queries along with the
    XML responses, and query results. As a side effect, all queries, XML responses, and results will
    be stored in an output directory for manual inspection as files.

    Parameters
    ----------
    topics : iterator of topic
        The topics that will serve as the source of the queries.
    webmias : WebMIaSIndex
        The index of a deployed WebMIaS Java Servlet that will be queried to retrieve the query
        results.
    positions : dict of (str, float)
        A map from paragraph identifiers to estimated positions of paragraphs in their parent
        documents. The positions are in the range [0; 1].
    estimates : sequence of float
        Estimates of P(relevant | position) in the form of a histogram.
    output_directory : Path or None, optional
        The path to a directore, where reranked results will be stored as files. When the
        output_directory is None, no files will be produced.
    num_workers : int, optional
        The number of processes that will send queries.

    Yields
    ------
    (MathFormat, Topic, sequence of ExecutedProcessedQuery)
        A format in which the mathematical formulae were represented in a query, and topics, each
        with in iterable of queries along with the XML responses and query results.
    """
    for topic in topics:
        assert isinstance(topic, Topic)
    assert isinstance(webmias, WebMIaSIndex)
    assert isinstance(num_workers, int)
    assert num_workers > 0

    LOGGER.info(
        "Using %d formats to represent mathematical formulae in queries:",
        len(Formula.math_formats))
    for math_format in sorted(Formula.math_formats):
        LOGGER.info("- %s", math_format)

    with Pool(num_workers) as pool:
        for math_format, topic, executed_queries in pool.imap_unordered(_query_webmias_helper, (
                (topic, math_format, webmias, output_directory)
                for topic in tqdm(topics, desc="get_results")
                for math_format in Formula.math_formats)):
            executed_processed_queries = []
            for executed_query in executed_queries:
                if output_directory:
                    executed_query.save(output_directory)
                executed_processed_queries.append(
                        ExecutedProcessedQuery.from_elements(executed_query, positions, estimates))
            yield(math_format, topic, executed_processed_queries)


def _rerank_and_merge_results_helper(args):
    math_format, topic, queries, output_directory, num_results = args
    results = []
    for aggregation in MIaSResult.aggregations:
        result_deques = []
        for query in queries:
            with query.use_aggregation(aggregation):
                if output_directory:
                    query.save(output_directory)
                result_deques.append(deque(query.results))
        result_list = []
        result_list_identifiers = set()
        for query, result_dequeue in cycle(zip(queries, result_deques)):
            if not sum(len(result_dequeue) for result_dequeue in result_deques):
                break  # All result deques are already empty, stop altogether
            if len(result_list) == num_results:
                break  # The result list is already full, stop altogether
            if not result_dequeue:
                continue  # The result deque for this query is already empty, try the next one
            try:
                for _ in range(query.stripe_width):
                    result = result_dequeue.popleft()
                    while result.identifier in result_list_identifiers:
                        result = result_dequeue.popleft()
                    result_list.append(result)
                    result_list_identifiers.add(result.identifier)
                    if len(result_list) == num_results:
                        break
            except IndexError:
                continue
        results.append((aggregation, math_format, topic, result_list))
    return results


def rerank_and_merge_results(
        results, identifiers, output_directory=None, num_workers=1,
        num_results=TARGET_NUMBER_OF_RESULTS):
    """
    Reranks results using position, probability, and density estimates produced by the NTCIR Math
    Density Estimator package, and merges them into final result lists. As a side effect, the
    reranked results as well as the final result lists will be stored in an output directory for
    manual inspection as files.

    Parameters
    ----------
    results : iterator of (MathFormat, Topic, sequence of ExecutedProcessedQuery)
        A format in which the mathematical formulae were represented in a query, and topics, each
        with in iterable of queries along with the XML responses and query results.
    identifiers : set of str, or KeysView of str
        A set of all paragraph identifiers in a dataset. When the target number of results for a
        topic cannot be met by merging the queries, the identifiers are randomly sampled.
    output_directory : Path or None, optional
        The path to a directore, where reranked results will be stored as files. When the
        output_directory is None, no files will be produced.
    num_workers : int, optional
        The number of processes that will rerank results.
    num_results : int, optional
        The target number of results for a topic.

    Yields
    ------
    (ScoreAggregationStrategy, MathFormat, ResultList)
        A score aggregation strategy that was used to rerank the results, a format in which the
        mathematical formulae were represented in a query, and a final result list.
    """
    assert isinstance(identifiers, (set, KeysView))
    assert output_directory is None or isinstance(output_directory, Path)
    assert isinstance(num_workers, int)
    assert num_workers > 0
    assert isinstance(num_results, int)
    assert num_results > 0
    assert len(identifiers) >= num_results, \
        "The target number of results for a topic is greater than the dataset size"

    final_results = dict()
    already_warned = set()
    artificial_results = [  # Take only num_results from identifiers, without creating a list
        ArtificialResult(identifier, -float("inf"))
        for identifier, _ in zip(identifiers, range(num_results))]

    LOGGER.info(
        "Using %d strategies to aggregate MIaS scores with probability estimates:",
        len(MIaSResult.aggregations))
    for aggregation in sorted(MIaSResult.aggregations):
        LOGGER.info("- %s", aggregation)
    if output_directory:
        LOGGER.info("Storing reranked per-query result lists in %s", output_directory)
    with Pool(num_workers) as pool:
        for merged_results in pool.imap_unordered(_rerank_and_merge_results_helper, (
                        (math_format, topic, queries, output_directory, num_results)
                        for math_format, topic, queries in tqdm(
                            results, desc="rerank_and_merge_results"))):
            for aggregation, math_format, topic, result_list in merged_results:
                if len(result_list) < num_results:
                    if topic not in already_warned:
                        LOGGER.warning(
                            "Result list for topic %s contains only %d / %d results, sampling "
                            "the dataset", topic, len(result_list), num_results)
                        already_warned.add(topic)
                    result_list.extend(artificial_results[:num_results - len(result_list)])
                assert len(result_list) == num_results

                if (aggregation, math_format) not in final_results:
                    final_results[(aggregation, math_format)] = []
                final_results[(aggregation, math_format)].append((topic, result_list))

    if output_directory:
        LOGGER.info("Storing final result lists in %s", output_directory)
    for (aggregation, math_format), topics_and_results in tqdm(final_results.items()):
        if output_directory:
            with (output_directory / Path(PATH_FINAL_RESULT % (
                    math_format.identifier, aggregation.identifier))).open("wt") as f:
                write_tsv(f, topics_and_results)
        for topic, result in topics_and_results:
            yield (aggregation, math_format, ResultList(topic, result))


def get_topics(input_file):
    """
    Reads topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format from an XML file.

    Note
    ----
    Topics are returneed in the order in which they appear in the XML file.

    Parameters
    ----------
    input_file : Path
        The path to an input file with topics.

    Returns
    -------
    iterable of Topic
        Topics from the XML file.
    """
    with input_file.open("rt") as f:
        topics = list(Topic.from_file(f))
    return topics


def get_webmias(url, index_number):
    """
    Establishes a connection with an index of a deployed WebMIaS Java Servlet.

    Parameters
    ----------
    url : ParseResult
        The URL at which a WebMIaS Java Servlet has been deployed.
    index_number : int, optional
        The numeric identifier of the WebMIaS index that corresponds to the dataset.

    Return
    ------
    WebMIaS
        A representation of the WebMIaS index.
    """
    webmias = WebMIaSIndex(url, index_number)
    return webmias

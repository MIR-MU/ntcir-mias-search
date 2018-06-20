"""
This module contains utility functions.
"""

from logging import getLogger
import gzip
import pickle

from lxml.etree import _Element, QName
from lxml.objectify import deannotate

LOGGER = getLogger(__name__)
MIN_RELEVANT_SCORE = 2.0
XPATH_NAMESPACED = "descendant-or-self::*[namespace-uri() != '']"


def remove_namespaces(tree):
    """
    Removes namespace declarations, and namespaces in element names from an XML tree.

    Parameters
    ----------
    tree : _Element
        The tree from which namespace declarations, and namespaces will be removed.
    """
    assert isinstance(tree, _Element)

    for element in tree.xpath(XPATH_NAMESPACED):
        element.tag = QName(element).localname
    deannotate(tree, cleanup_namespaces=True)


def write_tsv(output_file, topics_and_results):
    """
    Produces a tab-separated-value (TSV) file, each line containing a topic identifier, a reserved
    value, an identifier of a paragraph in a result, the rank of the result, the score of the
    result, the estimated position of the paragraph in the original document, the estimated
    probability of relevance of the result, and the relevance of the result according to relevance
    judgements.

    Parameters
    ----------
    output_file : file
        The output TSV file.
    topics_and_results : iterator of (Topic, iterable of Result)
        Topics, each with a sorted iterable of results.
    """
    for topic, results in topics_and_results:
        for rank, result in enumerate(results):
            output_file.write("%s\tRESERVED\t%s\t%d\t%s\n" % (
                topic.name, result.identifier, rank + 1, result))


def get_judgements(input_file, min_relevant_score=MIN_RELEVANT_SCORE):
    """
    Reads relevance judgements in the NTCIR-11 Math-2, and NTCIR-12 MathIR format from a text file.

    Parameters
    ----------
    input_file : file
        An input file with relevance judgements in the NTCIR-11 Math-2, and NTCIR-12 MathIR format.
    min_relevant_score : double
        Only relevance judgements with score greater than or equal to min_relevant_score are
        considered relevant.

    Returns
    -------
    dict of (str, dict of (str, bool))
        A map between NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR judgement identifiers,
        paragraph identifiers, and relevance judgements.
    """
    judgements = dict()
    for line in input_file:
        topic, _, identifier, score = line.split(' ')
        relevant = float(score) >= min_relevant_score
        if topic not in judgements:
            judgements[topic] = dict()
        judgements[topic][identifier] = relevant
    return judgements


def get_positions(input_file):
    """
    Reads estimates of paragraph positions produced by the NTCIR Math Density Estimator package.

    Parameters
    ----------
    input_file : file
        An input file with paragraph positions.

    Returns
    -------
    dict of (Path, dict of (str, float))
        A map between dataset paths, paragraph identifiers, and estimated paragraph positions in the
        original documents.
    """
    with gzip.open(input_file, "rb") as f:
        return pickle.load(f)


def get_estimates(input_file):
    """
    Reads density, and probability estimates from a output file produced by the NTCIR Math Density
    Estimator package.

    Parameters
    ----------
    input_file : file
        An input file with density, and probability estimates.

    Returns
    -------
    six-tuple of (sequence of float)
        Estimates of P(relevant), p(position), p(position | relevant), P(position, relevant), and
        P(relevant | position) in the form of histograms.

    dict of (Path, sequence of double)
        A map between paths
    """
    with gzip.open(input_file, "rb") as f:
        return pickle.load(f)

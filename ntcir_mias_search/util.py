"""
This module contains utility functions.
"""

from logging import getLogger
import gzip
import pickle

import numpy  # noqa:F401 Required to unpickle positions.pkl.gz

from lxml import etree
from lxml.etree import _Element, QName, XMLParser
from lxml.objectify import deannotate

LOGGER = getLogger(__name__)
MIN_RELEVANT_SCORE = 0.0
XPATH_NAMESPACED = "descendant-or-self::*[namespace-uri() != '']"
TOP_ITEMS_NUMBER = 5
BOTTOM_ITEMS_NUMBER = 5


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


def xml_documents_equal(first, second):
    """
    Tests two XML documents for equality.

    Parameters
    ----------
    first : str
        A string with the first XML document.
    second : str
        A string with the second XML document.

    Returns
    -------
    bool
        Whether the two documents are equal.
    """
    parser = XMLParser(encoding="utf-8", remove_blank_text=True)
    first_document = etree.fromstring(first, parser=parser)
    second_document = etree.fromstring(second, parser=parser)
    return xml_elements_equal(first_document, second_document)


def xml_elements_equal(first, second):
    """
    Tests two XML elements for equality.

    Parameters
    ----------
    first : _Element
        The first element.
    second : _Element
        The second element.

    Returns
    -------
    bool
        Whether the two elements are equal.
    """
    if first.tag != second.tag:
        return False
    if first.text != second.text:
        return False
    if first.tail != second.tail:
        return False
    if first.attrib != second.attrib:
        return False
    if len(first) != len(second):
        return False
    return all(xml_elements_equal(c1, c2) for c1, c2 in zip(first, second))


def write_tsv(output_file, topics_and_results):
    """
    Produces a tab-separated-value (TSV) file, each line containing a topic identifier, a reserved
    value, an identifier of a paragraph in a result, the rank of the result, the MIaS score of the
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
        Only relevance judgements with score greater than min_relevant_score are considered
        relevant.

    Returns
    -------
    dict of (str, dict of (str, bool))
        A map between NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR judgement identifiers,
        paragraph identifiers, and relevance judgements.
    """
    judgements = dict()
    for line in input_file:
        topic, _, identifier, score = line.split(' ')
        relevant = float(score) > min_relevant_score
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


def log_sequence(items):
    """
    Logs a sequence of values, only showing a couple to the user.

    Parameters
    ----------
    items : sequence of values
        The values that wlll be converted to string, and logged.
    """
    for item_number, item in enumerate(items):
        line = "- %s" % item
        if item_number < TOP_ITEMS_NUMBER or item_number > len(items) - BOTTOM_ITEMS_NUMBER - 1:
            LOGGER.info(line)
        else:
            if item_number == TOP_ITEMS_NUMBER:
                LOGGER.info("- ...")
            LOGGER.debug(line)

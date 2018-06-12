"""
This module contains utility functions.
"""

from logging import getLogger

from lxml.etree import _Element, QName
from lxml.objectify import deannotate

LOGGER = getLogger(__name__)
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
    value, an identifier of a paragraph in a result, the rank of the result, and the score of the
    result.

    Parameters
    ----------
    output_file : file
        The output TSV file.
    topics_and_results : iterator of (Topic, iterable of Result)
        Topics, each with a sorted iterable of results.
    """
    for topic, results in topics_and_results:
        for rank, result in enumerate(results):
            output_file.write("%s\tRESERVED\t%s\t%d\t%f\n" % (
                topic.name, result.identifier, rank + 1, result.aggregate_score()))

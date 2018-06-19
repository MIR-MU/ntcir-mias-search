"""
This module defines data types for accessing a deployed WebMIaS Java Servlet.
"""

from logging import getLogger
from urllib.parse import ParseResult

from lxml import etree
from lxml.etree import _Element, XMLParser
import requests


LOGGER = getLogger(__name__)
TARGET_NUMBER_OF_RESULTS = 1000


class WebMIaSIndex(object):
    """
    This class represents an index of a deployed WebMIaS Java Servlet.

    Parameters
    ----------
    url : ParseResult
        The URL at which a WebMIaS Java Servlet has been deployed.
    index_number : int, optional
        The numeric identifier of the WebMIaS index that corresponds to the dataset.

    Attributes
    ----------
    url : ParseResult
        The URL at which a WebMIaS Java Servlet has been deployed.
    index_number : int, optional
        The numeric identifier of the WebMIaS index that corresponds to the dataset.
    """
    def __init__(self, url, index_number=0):
        assert isinstance(url, ParseResult)
        assert index_number >= 0

        response = requests.get(url.geturl())
        assert response.status_code == 200

        self.url = url
        self.index_number = index_number

    def query(self, query):
        """
        Queries the WebMIaS index and returns a response XML document.

        Parameters
        ----------
        query : Query
            The query.

        Return
        ------
        _Element
            The response XML document.
        """
        response = requests.post("%s/ws/search" % self.url.geturl(), data={
            "limit": TARGET_NUMBER_OF_RESULTS,
            "index": self.index_number,
            "query": query.payload,
        })
        parser = XMLParser(encoding="utf-8", recover=True)
        response_element = etree.fromstring(response.content, parser=parser)
        assert isinstance(response_element, _Element)
        return response_element

    def __repr__(self):
        return "%s(%s, %d)" % (self.__class__.__name__, self.url.geturl(), self.index_number)

"""
These are the processing functions and data types for the NTCIR MIaS Search package.
"""

from abc import abstractmethod
from collections import deque, KeysView
from contextlib import contextmanager
from copy import deepcopy
from itertools import cycle
from logging import getLogger
from math import inf, log10
from multiprocessing import Pool
from pathlib import Path
import re
from urllib.parse import ParseResult

from lxml import etree
from lxml.etree import _Element, Element, QName, XMLParser
from lxml.objectify import deannotate
from numpy import linspace
from tqdm import tqdm
import requests


LOGGER = getLogger(__name__)
NAMESPACES = {
    "m": "http://www.w3.org/1998/Math/MathML",
    "mws": "http://search.mathweb.org/ns",
    "ntcir": "http://ntcir-math.nii.ac.jp/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}
PATH_QUERY = "%s_%s.%d.query.txt"
PATH_RESPONSE = "%s_%s.%d.response.xml"
PATH_RESULT = "%s_%s.%d.results.%s.tsv"
PATH_FINAL_RESULT = "final_%s.%s.tsv"
TARGET_NUMBER_OF_RESULTS = 1000
XPATH_NAMESPACED = "descendant-or-self::*[namespace-uri() != '']"
XPATH_PMATH = ".//m:annotation-xml[@encoding='MathML-Presentation']"
XPATH_TEX = ".//m:annotation[@encoding='application/x-tex']"


def remove_namespaces(tree):
    """
    Removes namespace declarations, and namespaces in element names from an XML tree.

    Parameters
    ----------
    tree : _Element
        The tree from which namespace declarations, and namespaces will be removed.
    """
    for element in tree.xpath(XPATH_NAMESPACED):
        element.tag = QName(element).localname
    deannotate(tree, cleanup_namespaces=True)


class NamedEntity(object):
    """
    This class represents an entity with a unique identifier, and a human-readable text description.

    Attributes
    ----------
    identifier : str
        A unique identifier.
    description : str
        A human-readable text description.
    """

    def __str__(self):
        return "%s (look for '%s' in filenames)" % (self.description, self.identifier)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.identifier, self.description)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return isinstance(other, NamedEntity) and self.identifier == other.identifier

    def __lt__(self, other):
        return isinstance(other, NamedEntity) and self.identifier < other.identifier


class Singleton(type):
    """
    This metaclass designates a class as a singleton. No more than one instance
    of the class will be instantiated.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MathFormat(NamedEntity, metaclass=Singleton):
    """
    This class represents a format in which mathematical formulae are represented.
    """
    @abstractmethod
    def encode(self, Formula):
        """
        Returns a text representation of a mathematical formula.

        Parameters
        ----------
        formula : Formula
            A mathematical formula that will be represented in this format.

        Returns
        -------
        str
            The text representation of the mathematical formula.
        """
        pass


class TeX(MathFormat):
    """
    This class represents the popular math authoring language of TeX.
    """
    def __init__(self):
        self.identifier = "TeX"
        self.description = "The TeX language by professor Knuth"

    def encode(self, formula):
        assert isinstance(formula, Formula)

        return formula.tex_text


class PMath(MathFormat):
    """
    This class represents the Presentation MathML XML language.
    """
    def __init__(self):
        self.identifier = "PMath"
        self.description = "Presentation MathML XML language"

    def encode(self, formula):
        assert isinstance(formula, Formula)

        return formula.pmath_text


class CMath(MathFormat):
    """
    This class represents the Content MathML XML language.
    """
    def __init__(self):
        self.identifier = "CMath"
        self.description = "Content MathML XML language"

    def encode(self, formula):
        assert isinstance(formula, Formula)

        return formula.cmath_text


class PCMath(MathFormat):
    """
    This class represents the combined Presentation, and Content MathML XML language.
    """
    def __init__(self):
        self.identifier = "PCMath"
        self.description = "Combined Presentation and Content MathML XML language"

    def encode(self, formula):
        assert isinstance(formula, Formula)

        return "%s %s" % (formula.pmath_text, formula.cmath_text)


class QueryExpansionStrategy(NamedEntity, metaclass=Singleton):
    """
    This class represents a query expansion strategy for extracting queries out of an NTCIR-10 Math,
    NTCIR-11 Math-2, and NTCIR-12 MathIR topic.
    """
    @abstractmethod
    def produce_queries(self, topic):
        """
        Produces triples of formulae, keywords, and stripe widths from a NTCIR-10 Math, NTCIR-11
        Math-2, and NTCIR-12 MathIR topic.

        Parameters
        ----------
        topic : Topic
            A topic that will serve as the source of the triples.

        Yield
        -----
        (sequence of Formula, sequence of str, int)
            Formulae, keywords, and stripe widths.
        """
        pass


class LeaveRightmostOut(QueryExpansionStrategy):
    """
    This class represents the Leave Rightmost Out (LRO) query expansion strategy (Růžička et al.,
    2014).
    """
    def __init__(self):
        self.identifier = "LRO"
        self.description = "Leave Rightmost Out strategy (Růžička et al. 2014)"
    
    def produce_queries(self, topic):
        assert isinstance(topic, Topic)

        num_queries = len(topic.formulae) + len(topic.keywords) + 1
        for query_index_keywords, first_keyword in enumerate(range(len(topic.keywords) + 1)):
            yield (
                topic.formulae, topic.keywords[first_keyword:], num_queries - query_index_keywords)
        for query_index_formulae, last_formula in enumerate(range(len(topic.formulae) - 1, -1, -1)):
            yield (
                topic.formulae[0:last_formula], topic.keywords,
                num_queries - len(topic.keywords) - query_index_formulae - 1)


class ScoreAggregationStrategy(NamedEntity):
    """
    This class represents a strategy for aggregating a real score, and a probability of relevance
    into a single aggregate score.
    """
    @abstractmethod
    def aggregate_score(self, result):
        """
        Aggregates a score assigned to a paragraph in a result by MIaS, and an estimated probability
        that the paragraph is relevant by the NTCIR Math Density Estimator package.

        Parameters
        ----------
        result : Result
            A result.

        Returns
        -------
        float
            The aggregate score of the result.
        """
        pass


class MIaSScore(ScoreAggregationStrategy, metaclass=Singleton):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into
    an aggregate score. The aggregate score corresponds to the score, the probability estimate is
    discarded.
    """
    def __init__(self):
        self.identifier = "orig"
        self.description = "The original score with the probability estimate discarded"

    def aggregate_score(self, result):
        assert isinstance(result, Result)

        score = result.score
        assert isinstance(score, float)

        return score


class LogGeometricMean(ScoreAggregationStrategy, metaclass=Singleton):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into the
    common logarithm of their geometric mean.
    """
    def __init__(self):
        self.identifier = "geom"
        self.description = "Log10 of the geometric mean"

    def aggregate_score(self, result):
        assert isinstance(result, Result)

        score = result.rescaled_score()
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0

        p_relevant = result.p_relevant
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        if score == 0.0 or p_relevant == 0.0:
            log_geometric_mean = -inf
        else:
            log_score = log10(score)
            log_p_relevant = log10(p_relevant)
            log_geometric_mean = (log_score - log_p_relevant) / 2.0
        return log_geometric_mean


class LogHarmonicMean(ScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into the
    common logarithm of their weighted harmonic mean.

    Parameters
    ----------
    alpha : float
        The weight of a probability estimate (the weight is in the range [0; 1]). The weight of a
        score is 1 - alpha.
    """
    def __init__(self, alpha):
        assert isinstance(alpha, float)
        assert alpha >= 0.0 and alpha <= 1.0

        self.identifier = "harm%0.1f" % alpha
        self.description = "Log10 of the weighted harmonic mean (alpha = %0.1f)" % alpha
        self.alpha = alpha

    def aggregate_score(self, result):
        assert isinstance(result, Result)

        score = result.rescaled_score()
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0

        p_relevant = result.p_relevant
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        if score == 0.0 or p_relevant == 0.0:
            log_harmonic_mean = -inf
        else:
            log_harmonic_mean = -log10((1 - self.alpha) / score + self.alpha / p_relevant)
        return log_harmonic_mean


class Formula(object):
    """
    This class represents a formula in a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic.

    Parameters
    ----------
    tex : str
        The TeX representation of the formula.
    pmath : _Element
        A {http://www.w3.org/1998/Math/MathML}math element containing a Presentation MathML
        representation of the formula.
    cmath : _Element
        A {http://www.w3.org/1998/Math/MathML}math element containing a Content MathML
        representation of the formula.

    Attributes
    ----------
    tex_tex : str
        The text of the TeX representation of the formula.
    pmath_text : str
        The text of the {http://www.w3.org/1998/Math/MathML}math element containing a Presentation
        MathML representation of the formula.
    cmath_text : str
        The text of the {http://www.w3.org/1998/Math/MathML}math element containing a Content MathML
        representation of the formula.
    """
    def __init__(self, tex, pmath, cmath):
        assert isinstance(tex, str)
        assert isinstance(pmath, _Element)
        assert isinstance(cmath, _Element)
        pmath_text = etree.tostring(pmath, pretty_print=True).decode("utf-8")
        assert isinstance(pmath_text, str)
        cmath_text = etree.tostring(cmath, pretty_print=True).decode("utf-8")
        assert isinstance(cmath_text, str)

        self.tex_text = "$%s$" % tex
        self.pmath_text = pmath_text
        self.cmath_text = cmath_text

    def from_element(formula_tree):
        """
        Extracts a formula from a {http://ntcir-math.nii.ac.jp/}formula XML element.

        Parameters
        ----------
        formula_tree : _Element
            A {http://ntcir-math.nii.ac.jp/}formula XML element.

        Returns
        -------
        Formula
            The extracted formula.
        """
        assert isinstance(formula_tree, _Element)

        tex_query_texts = formula_tree.xpath("%s/text()" % XPATH_TEX, namespaces=NAMESPACES)
        assert len(tex_query_texts) == 1
        tex_query_text = tex_query_texts[0]
        tex_query_text = re.sub(r"%([\n\r])", r"\1", tex_query_text)  # Remove % from ends of lines
        tex_query_text = re.sub(r"[\n\r]+", r"", tex_query_text)  # Join multiline TeX formulae
        tex_query_text = re.sub(r"\\qvar{(.).*?}", r" \1 ", tex_query_text)  # Make \qvar one letter

        pmath_trees = formula_tree.xpath(XPATH_PMATH, namespaces=NAMESPACES)
        assert len(pmath_trees) == 1
        pmath_tree = pmath_trees[0]
        pmath_query_tree = Element(QName(NAMESPACES["m"], "math"))
        for pmath_child in pmath_tree.getchildren():
            pmath_query_tree.append(deepcopy(pmath_child))
        for qvar_tree in pmath_query_tree.xpath(".//mws:qvar", namespaces=NAMESPACES):
            mi_tree = Element(QName(NAMESPACES["m"], "mi"))
            mi_tree.text = qvar_tree.get("name")[0]  # Reduce mws:qvar/@name to m:mi with one letter
            qvar_tree.getparent().replace(qvar_tree, mi_tree)
        remove_namespaces(pmath_query_tree)

        cmath_trees = formula_tree.xpath(".//m:semantics", namespaces=NAMESPACES)
        assert len(cmath_trees) == 1
        cmath_tree = cmath_trees[0]
        cmath_query_tree = Element(QName(NAMESPACES["m"], "math"))
        for cmath_child in cmath_tree.getchildren():
            cmath_query_tree.append(deepcopy(cmath_child))
        for qvar_tree in cmath_query_tree.xpath(".//mws:qvar", namespaces=NAMESPACES):
            mi_tree = Element(QName(NAMESPACES["m"], "ci"))
            mi_tree.text = qvar_tree.get("name")[0]  # Reduce mws:qvar/@name to m:ci with one letter
            qvar_tree.getparent().replace(qvar_tree, mi_tree)
        cmath_query_tree_tex_trees = cmath_query_tree.xpath(XPATH_TEX, namespaces=NAMESPACES)
        assert len(cmath_query_tree_tex_trees) == 1
        cmath_query_tree_tex_tree = cmath_query_tree_tex_trees[0]
        cmath_query_tree_tex_tree.getparent().remove(cmath_query_tree_tex_tree)  # Remove TeX
        cmath_query_tree_pmath_trees = cmath_query_tree.xpath(XPATH_PMATH, namespaces=NAMESPACES)
        assert len(cmath_query_tree_pmath_trees) == 1
        cmath_query_tree_pmath_tree = cmath_query_tree_pmath_trees[0]
        cmath_query_tree_pmath_tree.getparent().remove(cmath_query_tree_pmath_tree) # Remove PMath
        remove_namespaces(cmath_query_tree)

        return Formula(tex_query_text, pmath_query_tree, cmath_query_tree)

    math_formats = set((TeX(), PMath(), CMath(), PCMath()))

    def __repr__(self, math_format=TeX()):
        return "%s(%s)" % (self.__class__.__name__, math_format.encode(self))


class Topic(object):
    """
    This class represents an NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic.

    Parameters
    ----------
    name : str
        The identifier of the topic as specified in the {http://ntcir-math.nii.ac.jp/}num element.
    formulae : iterable of Formula
        One or more formulae from the topic as specified in the
        {http://www.w3.org/1998/Math/MathML}math elements.
    keywords : iterable of str
        One or more keywords from the topic as specified in the
        {http://ntcir-math.nii.ac.jp/}keyword elements.

    Attributes
    ----------
    name : str
        The identifier of the topic as specified in the {http://ntcir-math.nii.ac.jp/}num element.
    formulae : iterable of Formula
        One or more formulae from the topic as specified in the
        {http://www.w3.org/1998/Math/MathML}math elements.
    keywords : iterable of str
        One or more keywords from the topic as specified in the
        {http://ntcir-math.nii.ac.jp/}keyword elements.
    """
    def __init__(self, name, formulae, keywords):
        assert isinstance(name, str)
        for formula in formulae:
            assert isinstance(formula, Formula)
        for keyword in keywords:
            assert isinstance(keyword, str)

        self.name = name
        self.formulae = formulae
        self.keywords = keywords

    @staticmethod
    def from_element(topic_tree):
        """
        Extracts a topic from a {http://ntcir-math.nii.ac.jp/}topic XML element.

        Parameters
        ----------
        topic_tree : _Element
            A {http://ntcir-math.nii.ac.jp/}topic XML element.

        Returns
        -------
        Topic
            The extracted topic.
        """
        assert isinstance(topic_tree, _Element)

        names = topic_tree.xpath("ntcir:num/text()", namespaces=NAMESPACES)
        assert len(names) == 1
        name = names[0]

        formulae = [
            Formula.from_element(formula_tree)
            for formula_tree in topic_tree.xpath(".//ntcir:formula", namespaces=NAMESPACES)]
        assert len(formulae) > 0

        keywords = topic_tree.xpath(".//ntcir:keyword/text()", namespaces=NAMESPACES)
        assert len(keywords) > 0

        return Topic(name, formulae, keywords)

    def from_file(input_file):
        """
        Reads topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format from an XML
        file.

        Note
        ----
        Topics are yielded in the order in which they appear in the XML file.

        Parameters
        ----------
        input_file : file
            An input XML file containing topics.

        Yields
        ------
        Topic
            A topic from the XML file.
        """
        input_tree = etree.parse(input_file)
        for topic_tree in input_tree.xpath(".//ntcir:topic", namespaces=NAMESPACES):
            yield Topic.from_element(topic_tree)

        def __repr__(self):
            return "%s(%s)" % (self.__class__.__name__, self.name)

        def __hash__(self):
            return hash(name)
        
        def __eq__(self, other):
            return instanceof(other, Topic) and self.name == other.name

    def query(self, math_format, webmias):
        """
        Produces queries from the topic, queries a WebMIaS index, and returns the queries along with
        the XML responses, and query results.

        Parameters
        ----------
        math_format : MathFormat
            The format in which the mathematical formulae will be represented in a query.
        webmias : WebMIaSIndex
            The index of a deployed WebMIaS Java Servlet that will be queried to retrieve the query
            results.

        Yields
        ------
        Query
            An unfinalized query along with the XML responses, and query results.
        """
        assert isinstance(math_format, MathFormat)
        assert isinstance(webmias, WebMIaSIndex)

        for query in Query.from_topic(self, math_format, webmias):
            yield query

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Topic) and self.name == other.name

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "%s(%s, %d formulae, %d topics)" % (
            self.__class__.__name__, self.name, len(self.formulae), len(self.keywords))


class Query(object):
    """
    This class represents a query extracted from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12
    MathIR topic along with the query results.

    Parameters
    ----------
    topic : Topic
        The topic that will serve as the source of the query.
    math_format : MathFormat
        The format in which the mathematical formulae are represented in the query.
    webmias : WebMIaSIndex
        The index of a deployed WebMIaS Java Servlet. The index will be immediately queried to
        retrieve the query results.
    payload : str
        The text content of the query.
    query_number : int
        The number of the query among all queries extracted from the topic.
    stripe_width : int
        The number of results this query will contribute to the final result list each time the
        query gets its turn.

    Attributes
    ----------
    aggregation : ScoreAggregationStrategy
        The score aggregation strategy that will be used to compute the aggregate score of the query
        results. By default, this corresponds to MIaSScore(), i.e. no score aggregation will be
        performed. Change this attribute by using the use_aggregation context manager method.
    topic : Topic
        The topic that served as the source of the query.
    payload : str
        The text content of the query.
    math_format : MathFormat
        The format in which the mathematical formulae are represented in the query.
    query_number : int
        The number of the query among all queries extracted from the topic.
    stripe_width : int
        The stripe width, i.e. number of results this query will contribute to the final result list
        each time the query gets its turn.
    response_text : str
        The text of the XML response.
    results : iterable of Result
        The query results. After the Query object has been constructed, this iterable will be empty.
        Use the finalize method to obtain the results.
    """
    def __init__(self, topic, math_format, webmias, payload, query_number, stripe_width):
        assert isinstance(topic, Topic)
        assert isinstance(math_format, MathFormat)
        assert isinstance(webmias, WebMIaSIndex)
        assert isinstance(payload, str)
        assert isinstance(query_number, int)
        assert query_number > 0
        assert isinstance(stripe_width, int)
        assert stripe_width > 0

        response = webmias.query(payload)
        assert isinstance(response, _Element)
        response_text = etree.tostring(response, pretty_print=True).decode("utf-8")
        assert isinstance(response_text, str)
        results = [
            etree.tostring(result_tree).decode("utf-8")
            for result_tree in response.xpath(".//result")]

        self.aggregation = MIaSScore()
        self.topic = topic
        self.math_format = math_format
        self.payload = payload
        self.query_number = query_number
        self.stripe_width = stripe_width
        self.response_text = response_text
        self._results = results
        self.results = []

    def finalize(self, positions, estimates):
        """
        Uses information recorded by the Query object constructor to construct the query results.

        Parameters
        ----------
        positions : dict of (str, float)
            A map from paragraph identifiers to estimated positions of paragraphs in their parent
            documents. The positions are in the range [0; 1].
        estimates : sequence of float
        Estimates of P(relevant | position) in the form of a histogram.
        """
        assert "_results" in self.__dict__
        assert not self.results

        parser = XMLParser(encoding="utf-8", recover=True)

        self.results = [
            Result.from_element(
                self, etree.fromstring(result, parser=parser), positions, estimates)
            for result in self._results]
        del self._results

    @contextmanager
    def use_aggregation(self, aggregation):
        """
        Changes the score aggregation strategy, and sorts query results according to the aggregated
        scores for the duration of the context.

        Parameters
        ----------
        aggregation : ScoreAggregationStrategy
            The score aggregation strategy that will be used to compute the aggregate score of the
            query results for the duration of the context.
        """
        assert isinstance(aggregation, ScoreAggregationStrategy)

        original_aggregation = self.aggregation
        original_results = self.results
        self.aggregation = aggregation
        if aggregation != MIaSScore():
            self.results = sorted(self.results)
        yield
        self.aggregation = original_aggregation
        self.results = original_results

    query_expansions = set((LeaveRightmostOut(), ))

    @staticmethod
    def from_topic(topic, math_format, webmias, query_expansion=LeaveRightmostOut()):
        """
        Produces queries from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic.

        Parameters
        ----------
        topic : Topic
            A topic that will serve as the source of the queries.
        math_format : MathFormat
            The format in which the mathematical formulae will be represented in a query.
        webmias : WebMIaSIndex
            The index of a deployed WebMIaS Java Servlet. The index will be immediately queried to
            retrieve the query results.
        query_expansion : QueryExpansionStrategy
            A query expansion strategy that produces triples of formulae, keywords, and stripe
            widths.

        Yield
        -----
        Query
            A query produced from the topic.
        """
        assert isinstance(topic, Topic)
        assert isinstance(math_format, MathFormat)
        assert isinstance(query_expansion, QueryExpansionStrategy)
        assert isinstance(webmias, WebMIaSIndex)

        for query_number, (formulae, keywords, stripe_width) in \
                enumerate(query_expansion.produce_queries(topic)):
            for formula in formulae:
                assert isinstance(formula, Formula)
            for keyword in keywords:
                assert isinstance(keyword, str)
            assert stripe_width > 0

            payload_formulae = ' '.join(math_format.encode(formula) for formula in formulae)
            payload_keywords = ' '.join('"%s"' % keyword for keyword in keywords)
            payload = "%s %s" % (payload_keywords, payload_formulae)

            yield Query(topic, math_format, webmias, payload, query_number + 1, stripe_width)

    def save(self, output_directory):
        """
        Stores the text content of the query, the XML document with the response, and the results as
        files.

        Parameters
        ----------
        output_directory : Path
            The path to the directory, where the output files will be stored.
        """
        assert isinstance(output_directory, Path)

        with (output_directory / Path(PATH_QUERY % (
                self.topic.name, self.math_format.identifier, self.query_number))).open("wt") as f:
            f.write(self.payload)
        with (output_directory / Path(PATH_RESPONSE % (
                self.topic.name, self.math_format.identifier, self.query_number))).open("wt") as f:
            f.write(self.response_text)
        with (output_directory / Path(PATH_RESULT % (
                self.topic.name, self.math_format.identifier, self.query_number,
                self.aggregation.identifier))).open("wt") as f:
            write_tsv(f, [(self.topic, self.results)])


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

    def query(self, payload):
        """
        Queries the WebMIaS index and returns the response XML document.

        Parameters
        ----------
        payload : str
            The text content of the query.

        Return
        ------
        _Element
            The response XML document.
        """
        assert isinstance(payload, str)

        response = requests.post("%s/ws/search" % self.url.geturl(), data = {
            "limit": TARGET_NUMBER_OF_RESULTS,
            "index": self.index_number,
            "query": payload,
        })
        parser = XMLParser(encoding="utf-8", recover=True)
        response_element = etree.fromstring(response.content, parser=parser)
        assert isinstance(response_element, _Element)
        return response_element

    def __repr__(self):
        return "%s(%s, %d)" % (self.__class__.__name__, self.url.geturl(), self.index_number)


class Result(object):
    """
    This class represents the result of a query.

    Parameters
    ----------
    query : Query
        The query that produced the result.
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The score of the result.
    p_relevant : float
        The estimated probability of relevance of the result.

    Attributes
    ----------
    query : Query
        The query that produced the result.
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The MIaS score of the result.
    p_relevant : float
        The estimated probability of relevance of the paragraph in the result.
    """
    def __init__(self, query, identifier, score, p_relevant):
        assert isinstance(query, Query)
        assert isinstance(identifier, str)
        assert isinstance(score, float)
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        self.query = query
        self.identifier = identifier
        self.score = score
        self.p_relevant = p_relevant
        self._aggregate_scores = dict()

    @staticmethod
    def from_element(query, result_tree, positions, estimates):
        """
        Extracts a result from a result XML element in a WebMIaS response.

        Parameters
        ----------
        query : Query
            The query that produced the result.
        result_tree : _Element
            A result XML element.
        positions : dict of (str, float)
            A map from paragraph identifiers to estimated positions of paragraphs in their parent
            documents. The positions are in the range [0; 1].
        estimates : sequence of float
            Estimates of P(relevant | position) in the form of a histogram.

        Returns
        -------
        Result
            The extracted result.
        """
        assert isinstance(query, Query)
        assert isinstance(result_tree, _Element)

        identifier_path_trees = result_tree.xpath(".//path")
        assert len(identifier_path_trees) == 1
        identifier_path_tree = identifier_path_trees[0]
        identifier_path = Path(identifier_path_tree.text)
        identifier = identifier_path.stem
        
        score_trees = result_tree.xpath(".//info")
        assert len(score_trees) == 1
        score_tree = score_trees[0]
        score_match = re.match(r"\s*score\s*=\s*([-0-9.]*)", score_tree.text)
        assert score_match
        score = float(score_match.group(1))

        assert identifier in positions
        position = positions[identifier]
        assert isinstance(position, float)
        assert position >= 0.0 and position < 1.0

        p_relevant = estimates[int(position * len(estimates))]
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        return Result(query, identifier, score, p_relevant)

    def rescaled_score(self):
        """
        Linearly rescales the MIaS score of the result to the range [0; 1] by taking into account
        all results to the query that produced this result.

        Returns
        -------
        float
            The linearly rescaled MIaS score to the range [0; 1].
        """
        min_score = min(result.score for result in self.query.results)
        max_score = max(result.score for result in self.query.results)

        if max_score == min_score:
            rescaled_score = 1.0
        else:
            rescaled_score = (self.score - min_score) / (max_score - min_score)
        assert rescaled_score >= 0.0 and rescaled_score <= 1.0

        return rescaled_score

    def aggregate_score(self):
        """
        Aggregates the MIaS score of the result, and the estimated probability of relevance of the
        paragraph in the result using the aggregation strategy of the query that produced this
        result.
        """
        if self.query.aggregation not in self._aggregate_scores:
            aggregate_score = self.query.aggregation.aggregate_score(self)
            self._aggregate_scores[self.query.aggregation] = aggregate_score
        aggregate_score = self._aggregate_scores[self.query.aggregation]
        assert isinstance(aggregate_score, float)
        return aggregate_score
    
    def __lt__(self, other):
        return isinstance(other, Result) and self.aggregate_score() > other.aggregate_score()

    aggregation_strategies = set(
        [MIaSScore(), LogGeometricMean()] + \
        [LogHarmonicMean(alpha) for alpha in linspace(0, 1, 11)])

    def __getstate__(self):  # Do not serialize the aggregate score cache
        return (self.query, self.identifier, self.score, self.p_relevant)

    def __setstate__(self, state):
        self.query, self.identifier, self.score, self.p_relevant = state
        self._aggregate_scores = dict()

    def __repr__(self):
        return "%s(%s, %f, %f)" % (
            self.__class__.__name__, self.identifier, self.score, self.p_relevant)


class FakeResult(Result):
    """
    This class represents an artificially created result.

    Parameters
    ----------
    identifier : str
        The identifier of the paragraph in the result.
    aggregate_score : float
        The aggregate score of the result.

    Attributes
    ----------
    identifier : str
        The identifier of the paragraph in the result.
    _aggregate_score : float
        The aggregate score of the result.
    """
    def __init__(self, identifier, aggregate_score):
        assert isinstance(aggregate_score, float)
        assert isinstance(identifier, str)

        self._aggregate_score = aggregate_score
        self.identifier = identifier

    def aggregate_score(self):
        return self._aggregate_score

    def __getstate__(self):
        return (self.identifier, self._aggregate_score)

    def __setstate__(self, state):
        self.identifier, self._aggregate_score = state


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


def _get_results_helper(args):
    topic, math_format, webmias = args
    queries = list(topic.query(math_format, webmias))
    return (math_format, topic, queries)


def query_webmias(topics, webmias, positions, estimates, num_workers=1):
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
    num_workers : int, optional
        The number of processes that will send queries.

    Yields
    ------
    (MathFormat, Topic, sequence of Query)
        A format in which the mathematical formulae were represented in a query, and topics, each
        with in iterable of queries along with the XML responses and query results.
    """
    for topic in topics:
        assert isinstance(topic, Topic)
    assert isinstance(webmias, WebMIaSIndex)
    assert isinstance(num_workers, int)
    assert num_workers > 0

    result = dict()
    LOGGER.info(
        "Using %d formats to represent mathematical formulae in queries:",
        len(Formula.math_formats))
    for math_format in sorted(Formula.math_formats):
        LOGGER.info("- %s", math_format)

    with Pool(num_workers) as pool:
        for math_format, topic, queries in pool.imap_unordered(_get_results_helper, (
#       for math_format, topic, queries in (_get_results_helper(args) for args in tqdm([
                (topic, math_format, webmias)
                for topic in tqdm(topics, desc="get_results")
                for math_format in Formula.math_formats)):
            for query in queries:
                query.finalize(positions, estimates)
            yield(math_format, topic, queries)


def _rerank_and_merge_results_helper(args):
    math_format, topic, queries, output_directory, num_results = args
    results = []
    for aggregation in Result.aggregation_strategies:
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
    Density Estimator package. As a side effect, the reranked results will be stored in an output
    directory for manual inspection as files.

    Parameters
    ----------
    results : iterator of (MathFormat, Topic, sequence of Query)
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
    (ScoreAggregationStrategy, MathFormat, Topic, sequence of Query)
        A score aggregation strategy that was used to rerank the results, a format in which the
        mathematical formulae were represented in a query, and topics, each with in iterable of
        queries along with the XML responses and query results.
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
    artificial_results = [  # Take only num_results from identifiers, without making them a list
        FakeResult(identifier, -inf) for identifier, _ in zip(identifiers, range(num_results))]

    LOGGER.info(
        "Using %d strategies to aggregate MIaS scores with probability estimates:",
        len(Result.aggregation_strategies))
    for aggregation in sorted(Result.aggregation_strategies):
        LOGGER.info("- %s", aggregation)
    if output_directory:
        LOGGER.info("Storing reranked per-query result lists in %s", output_directory)
    with Pool(num_workers) as pool:
        for merged_results in pool.imap_unordered(_rerank_and_merge_results_helper, (
                        (math_format, topic, queries, output_directory, num_results)
                        for math_format, topic, queries in tqdm(results,
                            desc="rerank_and_merge_results"))):
            for aggregation, math_format, topic, result_list in merged_results:
                if len(result_list) < num_results:
                    if topic not in already_warned:
                        LOGGER.warning(
                            "Result list for topic %s contains only %d / %d results, sampling",
                            topic, len(result_list), num_results)
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
        yield ((aggregation, math_format), topics_and_results)


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

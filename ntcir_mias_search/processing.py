"""
These are the processing functions and data types for the NTCIR MIaS Search package.
"""

from abc import abstractmethod
from collections import deque, KeysView
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


class MathFormat(NamedEntity):
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


class QueryExpansionStrategy(NamedEntity):
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
        for query_index_keywords, last_keyword in enumerate(range(len(topic.keywords), -1, -1)):
            yield (
                topic.formulae, topic.keywords[0:last_keyword], num_queries - query_index_keywords)
        for query_index_formulae, last_formula in enumerate(range(len(topic.formulae) - 1, -1, -1)):
            yield (
                topic.formulae[0:last_formula], topic.keywords,
                num_queries - query_index_keywords - query_index_formulae)


class ScoreAggregationStrategy(NamedEntity):
    """
    This class represents a strategy for aggregating a real score, and a probability of relevance
    into a single aggregate score.
    """
    def aggregate_score(self, estimated_result, query):
        """
        Aggregates a score assigned to a paragraph in a result by MIaS, and an estimated probability
        that the paragraph is relevant by the NTCIR Math Density Estimator package.

        Parameters
        ----------
        estimated_result : EstimatedResult
            A query result along with a score assigned by MIaS, and a relevance probability
            estimate.
        query : Query
            The query that produced the result.

        Returns
        -------
        float
            The aggregate score.
        """
        assert isinstance(estimated_result, EstimatedResult)
        assert isinstance(query, Query)
        assert query.aggregation == MIaSScore()
        assert estimated_result in query.results

        min_score = min(estimated_result.result.score for estimated_result in query.results)
        max_score = max(estimated_result.result.score for estimated_result in query.results)

        if max_score == min_score:
            score = 1.0
        else:
            score = (estimated_result.result.score - min_score) / (max_score - min_score)
        assert score >= 0.0 and score <= 1.0

        p_relevant = estimated_result.p_relevant
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        aggregate_score = self._aggregate_score(score, p_relevant)
        return aggregate_score

    @abstractmethod
    def _aggregate_score(self, score, p_relevant):
        """
        Aggregates a score assigned to a paragraph in a result by MIaS, and an estimated probability
        that the paragraph is relevant by the NTCIR Math Density Estimator package.

        Parameters
        ----------
        score : float
            The score assigned to a paragraph by MIaS linearly rescaled to the range [0; 1] by
            taking into account all the results to a query.
        p_relevant : float
            The estimated probability that the paragraph is relevant by the NTCIR Density Estimator
            Package.

        Returns
        -------
        float
            The aggregate score.
        """


class MIaSScore(ScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into the
    an aggregate score. The aggregate score corresponds to the score, the probability estimate is
    discarded.
    """
    def __init__(self):
        self.identifier = "orig"
        self.description = "The original score with the probability estimate discarded"

    def aggregate_score(self, result, *args):
        assert isinstance(result, Result)

        return result.score


class LogGeometricMean(ScoreAggregationStrategy):
    """
    This class represents a strategy for aggregating a score, and a probability estimate into the
    common logarithm of their geometric mean.
    """
    def __init__(self):
        self.identifier = "geom"
        self.description = "Log10 of the geometric mean"

    def _aggregate_score(self, score, p_relevant):
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0
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

    def _aggregate_score(self, score, p_relevant):
        assert isinstance(score, float)
        assert score >= 0.0 and score <= 1.0
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
        Reads topics in the NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR format.

        Note
        ----
        Topics are yielded in the order they appear in the XML file.

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

    def query(self, math_format, webmias, output_directory=None):
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
        output_directory : Path or None, optional
            The path to a directore, where all queries, XML responses, and results will be stored as
            files. When the output_directory is None, no files will be produced.

        Yields
        ------
        Query
            A query along with the XML responses, and query results.
        """
        assert isinstance(math_format, MathFormat)
        assert isinstance(webmias, WebMIaSIndex)
        assert output_directory is None or isinstance(output_directory, Path)

        for query in Query.from_topic(self, math_format, webmias):
            if output_directory:
                query.save(output_directory)
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
        The score aggregation strategy that was used to rerank the results. By default, this
        corresponds to MIaSScore(), i.e. no score aggregation strategy was used. Change this
        attribute after you have aggregated the scores in the query results using some different
        strategy.
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
        The query results.
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
        results = [Result.from_element(result_tree) for result_tree in response.xpath(".//result")]
        for result in results:
            assert isinstance(result, (EstimatedResult, Result))

        self.aggregation = MIaSScore()
        self.topic = topic
        self.math_format = math_format
        self.payload = payload
        self.query_number = query_number
        self.stripe_width = stripe_width
        self.response_text = response_text
        self.results = results

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
        The numeric identifier of the WebMIaS index that corresponds to the dataset. Defaults to
        %(default)d.

    Attributes
    ----------
    url : ParseResult
        The URL at which a WebMIaS Java Servlet has been deployed.
    index_number : int, optional
        The numeric identifier of the WebMIaS index that corresponds to the dataset. Defaults to
        %(default)d.
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
    identifier : str
        The identifier of the paragraph in the result.
    score : float
        The score of the result.
    """
    def __init__(self, identifier, score):
        assert isinstance(identifier, str)
        assert isinstance(score, float)

        self.identifier = identifier
        self.score = score

    @staticmethod
    def from_element(result_tree):
        """
        Extracts a result from a result XML element in a response from WebMIaS.

        Parameters
        ----------
        result_tree : _Element
            A result XML element.

        Returns
        -------
        Result
            The extracted result.
        """
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

        return Result(identifier, score)

    def __lt__(self, other):
        return isinstance(other, Result) and self.score > other.score

    aggregation_strategies = set(
        [LogGeometricMean()] + [LogHarmonicMean(alpha) for alpha in linspace(0, 1, 11)])


class EstimatedResult(object):
    """
    This class represents the result of a query along with its relevance probability estimate.

    Parameters
    ----------
    result : Result
        A query result.
    p_relevant : float
        The relevance probability estimate of the result.

    Attributes
    ----------
    result : Result
        A query result.
    p_relevant : float
        The relevance probability estimate of the result.
    """
    def __init__(self, result, p_relevant):
        assert isinstance(result, Result)
        assert isinstance(p_relevant, float)
        assert p_relevant >= 0 and p_relevant <= 1

        self.result = result
        self.p_relevant = p_relevant

    @staticmethod
    def from_positions_and_estimates(result, positions, estimates):
        """
        Extracts a relevance probability estimate from position, probability, and density estimates
        produced by the NTCIR Math Density Estimator package.

        Parameters
        ----------
        result : Result
            A query result.
        positions : dict of (str, float)
            A map from paragraph identifiers to estimated positions of paragraphs in their parent
            documents. The positions are in the range [0; 1].
        estimates : sequence of float
            Estimates of P(relevant | position) in the form of a histogram.

        Returns
        -------
        EstimatedResult
            The result of a query along with its relevance probability estimate.
        """
        assert isinstance(result, Result)

        position = positions[result.identifier]
        assert position >= 0.0 and position < 1.0

        p_relevant = estimates[int(position * len(estimates))]
        assert p_relevant >= 0.0 and p_relevant <= 1.0

        estimated_result = EstimatedResult(result, p_relevant)
        return estimated_result

    def __lt__(self, other):
        return isinstance(other, EstimatedResult) and self.result < other.result


def write_tsv(output_file, topics_and_results):
    """
    Produces a tab-separated-value (TSV) file, each line containing a topic identifier, a reserved
    value, an identifier of a paragraph in a result, the rank of the result, and the score of the
    result.

    Parameters
    ----------
    output_file : file
        The output TSV file.
    topics_and_results : iterable of (Topic, iterable of Result)
        Topics, each with an iterable of results.
    """
    for topic, results in topics_and_results:
        for rank, result in enumerate(sorted(results)):
            output_file.write("%s\tRESERVED\t%s\t%d\t%f\n" % (
                topic.name, result.identifier, rank + 1, result.score))


def _get_results_helper(args):
    topic, math_format, webmias, output_directory = args
    queries = list(topic.query(math_format, webmias, output_directory))
    return (math_format, topic, queries)


def get_results(topics, webmias, output_directory=None, num_workers=1):
    """
    Produces queries from topics, queries a WebMIaS index, and returns the queries along with the
    XML responses, and query results. As a side effect, all queries, XML responses, and results will
    be stored in an output directory for manual inspection as files.

    Parameters
    ----------
    topics : iterable of topic
        The topics that will serve as the source of the queries.
    webmias : WebMIaSIndex
        The index of a deployed WebMIaS Java Servlet that will be queried to retrieve the query
        results.
    output_directory : Path or None, optional
        The path to a directore, where all queries, XML responses, and results will be stored as
        files. When the output_directory is None, no files will be produced.
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
    assert output_directory is None or isinstance(output_directory, Path)
    assert isinstance(num_workers, int)
    assert num_workers > 0

    result = dict()
    LOGGER.info(
        "Using %d formats to represent mathematical formulae in queries:",
        len(Formula.math_formats))
    for math_format in sorted(Formula.math_formats):
        LOGGER.info("- %s", math_format)
    if output_directory:
        LOGGER.info(
            "Storing queries, responses, and per-query result lists in %s", output_directory)

    with Pool(num_workers) as pool:
        for math_format, topic, queries in pool.imap_unordered(_get_results_helper, (
#       for math_format, topic, queries in (_get_results_helper(args) for args in tqdm([
                (topic, math_format, webmias, output_directory)
                for topic in tqdm(topics, desc="get_results")
                for math_format in Formula.math_formats)):
            yield(math_format, topic, queries)


def get_estimates(queries, positions, estimates):
    """
    Inserts relevance probability estimates into query results.

    Parameters
    ----------
    sequence of Query
        Queries containing results without relevance probability estimates.
    positions : dict of (str, float)
        A map from paragraph identifiers to estimated positions of paragraphs in their parent
        documents. The positions are in the range [0; 1].
    estimates : sequence of float
        Estimates of P(relevant | position) in the form of a histogram.

    Returns
    -------
    sequence of Query
        Queries containing results with relevance probability estimates.
    """
    estimated_queries = deepcopy(queries)
    for query, estimated_query in zip(queries, estimated_queries):
        estimated_query.results = [
            EstimatedResult.from_positions_and_estimates(result, positions, estimates)
            for result in query.results]
    return estimated_queries


def _rerank_results_helper(args):
    math_format, topic, queries, estimated_queries, output_directory = args
    results = [(MIaSScore(), math_format, topic, queries)]
    for aggregation in Result.aggregation_strategies:
        reranked_queries = deepcopy(queries)
        for estimated_query, reranked_query in zip(estimated_queries, reranked_queries):
            for estimated_result, reranked_result in zip(
                    estimated_query.results, reranked_query.results):
                reranked_result.score = aggregation.aggregate_score(
                    estimated_result, estimated_query)
            reranked_query.aggregation = aggregation
            if output_directory:
                reranked_query.save(output_directory)
        results.append((aggregation, math_format, topic, reranked_queries))
    return results


def rerank_results(results, positions, estimates, output_directory=None, num_workers=1):
    """
    Reranks results using position, probability, and density estimates produced by the NTCIR Math
    Density Estimator package. As a side effect, the reranked results will be stored in an output
    directory for manual inspection as files.

    Parameters
    ----------
    results : iterable of (MathFormat, Topic, sequence of Query)
        A format in which the mathematical formulae were represented in a query, and topics, each
        with in iterable of queries along with the XML responses and query results.
    positions : dict of (str, float)
        A map from paragraph identifiers to estimated positions of paragraphs in their parent
        documents. The positions are in the range [0; 1].
    estimates : sequence of float
        Estimates of P(relevant | position) in the form of a histogram.
    output_directory : Path or None, optional
        The path to a directore, where reranked results will be stored as files. When the
        output_directory is None, no files will be produced.
    num_workers : int, optional
        The number of processes that will rerank results.

    Yields
    ------
    (ScoreAggregationStrategy, MathFormat, Topic, sequence of Query)
        A score aggregation strategy that was used to rerank the results, a format in which the
        mathematical formulae were represented in a query, and topics, each with in iterable of
        queries along with the XML responses and query results.
    """
    assert output_directory is None or isinstance(output_directory, Path)
    assert isinstance(num_workers, int)
    assert num_workers > 0

    LOGGER.info(
        "Using %d strategies to aggregate MIaS scores with probability estimates:",
        len(Result.aggregation_strategies))
    for aggregation in sorted(Result.aggregation_strategies):
        LOGGER.info("- %s", aggregation)
    if output_directory:
        LOGGER.info("Storing reranked per-query result lists in %s", output_directory)

    with Pool(num_workers) as pool:
        for reranked_results in pool.imap_unordered(_rerank_results_helper, (
                        (
                            math_format, topic, queries,
                            get_estimates(queries, positions, estimates), output_directory)
                        for math_format, topic, queries in tqdm(results, desc="rerank_results"))):
            for aggregation, math_format, topic, reranked_queries in reranked_results:
                yield (aggregation, math_format, topic, reranked_queries)


def _merge_results_helper(args):
    aggregation, math_format, topic, reranked_queries, num_results = args
    result_dequeues = [deque(sorted(query.results)) for query in reranked_queries]
    result_list = []
    result_list_identifiers = set()
    for query, result_dequeue in cycle(zip(reranked_queries, result_dequeues)):
        if not sum(len(result_dequeue) for result_dequeue in result_dequeues):
            break  # All result dequeues are already empty, stop altogether
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
    return aggregation, math_format, topic, result_list


def merge_results(
        reranked_results, identifiers, output_directory=None, num_workers=1,
        num_results=TARGET_NUMBER_OF_RESULTS):
    """
    Merges reranked results into final result lists.

    Parameters
    ----------
    reranked_results : iterable of (ScoreAggregationStrategy, MathFormat, Topic, sequence of Query)
        A score aggregation strategy that was used to rerank the results, a format in which the
        mathematical formulae were represented in a query, and topics, each with in iterable of
        queries along with the XML responses and query results.
    identifiers : set of str, or KeysView of str
        A set of all paragraph identifiers in a dataset. When the target number of results for a
        topic cannot be met by merging the queries, the identifiers are randomly sampled.
    output_directory : Path or None, optional
        The path to a directore, where reranked results will be stored as files. When the
        output_directory is None, no files will be produced.
    num_workers : int, optional
        The number of processes that will merge results.
    num_results : int, optional
        The target number of results for a topic.
    
    Returns
    -------
    ((ScoreAggregationStrategy, MathFormat), iterable of (Topic, iterable of Result))
        A score aggregation strategy that was used to rerank the results, a format in which the
        mathematical formulae were represented in a query, and topics, each with an iterable of
        results.
    """
    assert isinstance(identifiers, (set, KeysView))
    assert output_directory is None or isinstance(output_directory, Path)
    assert isinstance(num_results, int)
    assert num_results > 0
    assert len(identifiers) >= num_results, \
        "The target number of results for a topic is greater than the dataset size"
    assert isinstance(num_workers, int)
    assert num_workers > 0

    final_results = dict()
    already_warned = set()
    artificial_results = [  # Take only num_results from identifiers, without making them a list
        Result(identifier, -inf) for identifier, _ in zip(identifiers, range(num_results))]
    with Pool(num_workers) as pool:
        for aggregation, math_format, topic, result_list in pool.imap(
                _merge_results_helper, (
                    (aggregation, math_format, topic, reranked_queries, num_results)
                    for aggregation, math_format, topic, reranked_queries
                    in tqdm(reranked_results, desc="merge_results"))):
            if len(result_list) < num_results:
                if topic not in already_warned:
                    LOGGER.warning(
                        "Result list for topic %s contains only %d / %d results, sampling randomly",
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

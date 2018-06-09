"""
These are the processing functions and data types for the MIaS Search package.
"""

from copy import deepcopy
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
import re
from urllib.parse import ParseResult

from lxml import etree
from lxml.etree import _Element, Element, QName, XMLParser
from lxml.objectify import deannotate
from tqdm import tqdm
import requests


LOGGER = getLogger(__name__)
NAMESPACES = {
    "m": "http://www.w3.org/1998/Math/MathML",
    "mws": "http://search.mathweb.org/ns",
    "ntcir": "http://ntcir-math.nii.ac.jp/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}
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

    math_formats = set(("TeX", "PMath", "CMath", "PCMath"))

    def as_text(self, math_format="TeX"):
        """
        Returns a text representation of the formula.

        Parameters
        ----------
        math_format : str
            The text format in which the formula will be represented. The format can be either TeX
            (TeX), Presentation MathML (PMath), Content MathML (CMath), or combined Presentation and
            Content MathML (PCMath).

        Returns
        -------
        str
            The text representation of the formula.
        """
        assert isinstance(math_format, str)
        assert math_format in self.math_formats

        if math_format == "TeX":
            text = self.tex_text
        elif math_format == "PMath":
            text = self.pmath_text
        elif math_format == "CMath":
            text = self.cmath_text
        elif math_format == "PCMath":
            text = "%s %s" % (self.pmath_text, self.cmath_text)

        return text

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.as_text())


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
        math_format : str
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
        assert isinstance(math_format, str)
        assert math_format in Formula.math_formats
        assert isinstance(webmias, WebMIaSIndex)
        assert output_directory is None or isinstance(output_directory, Path)

        for query in Query.from_topic(self, math_format, webmias):
            if output_directory:
                query.save(output_directory)
            yield query


def leave_rightmost_out(topic):
    """
    Produces triples of formulae, keywords, and stripe widths from a NTCIR-10 Math, NTCIR-11 Math-2,
    and NTCIR-12 MathIR topic. The triples are produced using the Leave Rightmost Out (LRO) strategy
    (Růžička et al., 2014).

    Parameters
    ----------
    topic : Topic
        A topic that will serve as the source of the triples.

    Yield
    -----
    (sequence of Formula, sequence of str, int)
        Formulae, keywords, and stripe widths.
    """
    assert isinstance(topic, Topic)

    num_queries = len(topic.formulae) + len(topic.keywords) + 1
    for query_index_keywords, last_keyword in enumerate(range(len(topic.keywords), -1, -1)):
        yield (
            topic.formulae, topic.keywords[0:last_keyword], num_queries - query_index_keywords)
    for query_index_formulae, last_formula in enumerate(range(len(topic.formulae) - 1, -1, -1)):
        yield (
            topic.formulae[0:last_formula], topic.keywords,
            num_queries - query_index_keywords - query_index_formulae)


class Query(object):
    """
    This class represents a query extracted from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12
    MathIR topic along with the query results.

    Parameters
    ----------
    topic : Topic
        The topic that will serve as the source of the query.
    math_format : str
        Whether the queries use TeX (TeX), Presentation MathML (PMath), Content MathML (CMath), or
        combined Presentation and Content MathML (PCMath) to represent mathematical formulae.
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
    topic : Topic
        The topic that served as the source of the query.
    payload : str
        The text content of the query.
    math_format : str
        The format in which the mathematical formulae will be represented in the query.
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
        assert isinstance(math_format, str)
        assert math_format in Formula.math_formats
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
            assert isinstance(result, Result)

        self.topic = topic
        self.math_format = math_format
        self.payload = payload
        self.query_number = query_number
        self.stripe_width = stripe_width
        self.response_text = response_text
        self.results = results

    @staticmethod
    def from_topic(topic, math_format, webmias, get_formulae_and_keywords=leave_rightmost_out):
        """
        Produces queries from a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic.

        Parameters
        ----------
        topic : Topic
            A topic that will serve as the source of the queries.
        math_format : str
            The format in which the mathematical formulae will be represented in a query.
        webmias : WebMIaSIndex
            The index of a deployed WebMIaS Java Servlet. The index will be immediately queried to
            retrieve the query results.
        get_formulae_and_keywords : callable
            A strategy that takes a NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR topic, and
            produces an iterable of triples of formulae, keywords, and stripe widths.

        Yield
        -----
        Query
            A query produced from the topic.
        """
        assert isinstance(topic, Topic)
        assert isinstance(math_format, str)
        assert math_format in Formula.math_formats
        assert callable(get_formulae_and_keywords)
        assert isinstance(webmias, WebMIaSIndex)

        for query_number, (formulae, keywords, stripe_width) in \
                enumerate(get_formulae_and_keywords(topic)):
            for formula in formulae:
                assert isinstance(formula, Formula)
            for keyword in keywords:
                assert isinstance(keyword, str)
            assert stripe_width > 0

            payload_formulae = ' '.join(formula.as_text(math_format) for formula in formulae)
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

        with (output_directory / Path("%s_%s.%d.query.txt" % (
                self.topic.name, self.math_format, self.query_number))).open("wt") as f:
            f.write(self.payload)
        with (output_directory / Path("%s_%s.%d.response.xml" % (
                self.topic.name, self.math_format, self.query_number))).open("wt") as f:
            f.write(self.response_text)
        with (output_directory / Path("%s_%s.%d.results.tsv" % (
                self.topic.name, self.math_format, self.query_number))).open("wt") as f:
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
        return "%s(%s, index_number=%d)" % (
            self.__class__.__name__, self.url.geturl(), self.index_number)


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


def write_tsv(output_file, topics_and_results):
    """
    Produces a tab-separated-value (TSV) file, each line containing a topic identifier, a reserved
    value, an identifier of a paragraph in a result, and the rank of the result.

    Parameters
    ----------
    output_file : file
        The output TSV file.
    topics_and_results : iterable of (Topic, iterable of Result)
        Topics, each with an iterable of results.
    """
    for topic, results in topics_and_results:
        for rank, result in enumerate(sorted(results)):
            output_file.write("%s\tRESERVED\t%s\t%d\n" % (topic.name, result.identifier, rank + 1))


def _query_webmias_helper(args):
    topic, math_format, webmias, output_directory = args
    queries = list(topic.query(math_format, webmias, output_directory))
    return (math_format, topic, queries)


def query_webmias(topics, webmias, output_directory=None, num_workers=1):
    """
    Produces queries from topics, queries a WebMIaS index, and returns the queries along with the
    XML responses, and query results.

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
        The number of processes that will perform the queries.

    Returns
    -------
    dict of (str, dict of (Topic, sequence of Query))
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
    with Pool(num_workers) as pool:
        for math_format, topic, queries in pool.imap_unordered(_query_webmias_helper, tqdm([
#       for math_format, topic, queries in (_query_webmias_helper(args) for args in tqdm([
                (topic, math_format, webmias, output_directory)
                for topic in topics for math_format in Formula.math_formats])):
            if math_format not in result:
                result[math_format] = dict()
            result[math_format][topic] = queries
    return result

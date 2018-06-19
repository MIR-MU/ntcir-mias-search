"""
This module defines data types for representing topics in the NTCIR-10 Math, NTCIR-11 Math-2, and
NTCIR-12 MathIR format.
"""

from copy import deepcopy
from logging import getLogger
import re

from lxml import etree
from lxml.etree import _Element, Element, QName

from .abstract import MathFormat
from .query import Query
from .util import remove_namespaces


LOGGER = getLogger(__name__)
NAMESPACES = {
    "m": "http://www.w3.org/1998/Math/MathML",
    "mws": "http://search.mathweb.org/ns",
    "ntcir": "http://ntcir-math.nii.ac.jp/",
    "xhtml": "http://www.w3.org/1999/xhtml",
}
XPATH_PMATH = ".//m:annotation-xml[@encoding='MathML-Presentation']"
XPATH_TEX = ".//m:annotation[@encoding='application/x-tex']"


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
    tex_text : str
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
        cmath_query_tree_tex_tree.getparent().remove(cmath_query_tree_tex_tree)   # Remove TeX
        cmath_query_tree_pmath_trees = cmath_query_tree.xpath(XPATH_PMATH, namespaces=NAMESPACES)
        assert len(cmath_query_tree_pmath_trees) == 1
        cmath_query_tree_pmath_tree = cmath_query_tree_pmath_trees[0]
        cmath_query_tree_pmath_tree.getparent().remove(cmath_query_tree_pmath_tree)  # Remove PMath
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
    judgements : dict of (str, bool)
        A map paragraph identifiers, and relevance judgements.

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
    judgements : dict of (str, bool)
        A map paragraph identifiers, and relevance judgements.
    """
    def __init__(self, name, formulae, keywords, judgements):
        assert isinstance(name, str)
        for formula in formulae:
            assert isinstance(formula, Formula)
        for keyword in keywords:
            assert isinstance(keyword, str)

        self.name = name
        self.formulae = formulae
        self.keywords = keywords
        self.judgements = judgements

    @staticmethod
    def from_element(topic_tree, judgements):
        """
        Extracts a topic from a {http://ntcir-math.nii.ac.jp/}topic XML element.

        Parameters
        ----------
        topic_tree : _Element
            A {http://ntcir-math.nii.ac.jp/}topic XML element.
        judgements : dict of (str, dict of (str, bool))
            A map between NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR judgement identifiers,
            paragraph identifiers, and relevance judgements.

        Returns
        -------
        Topic
            The extracted topic.
        """
        assert isinstance(topic_tree, _Element)

        names = topic_tree.xpath("ntcir:num/text()", namespaces=NAMESPACES)
        assert len(names) == 1
        name = names[0]
        assert isinstance(name, str)
        assert name in judgements

        formulae = [
            Formula.from_element(formula_tree)
            for formula_tree in topic_tree.xpath(".//ntcir:formula", namespaces=NAMESPACES)]
        assert len(formulae) > 0

        keywords = topic_tree.xpath(".//ntcir:keyword/text()", namespaces=NAMESPACES)
        assert len(keywords) > 0

        return Topic(name, formulae, keywords, judgements[name])

    def from_file(input_file, judgements):
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
        judgements : dict of (str, dict of (str, bool))
            A map between NTCIR-10 Math, NTCIR-11 Math-2, and NTCIR-12 MathIR judgement identifiers,
            paragraph identifiers, and relevance judgements.

        Yields
        ------
        Topic
            A topic from the XML file.
        """
        input_tree = etree.parse(input_file)
        for topic_tree in input_tree.xpath(".//ntcir:topic", namespaces=NAMESPACES):
            yield Topic.from_element(topic_tree, judgements)

        def __repr__(self):
            return "%s(%s)" % (self.__class__.__name__, self.name)

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return isinstance(other, Topic) and self.name == other.name

    def get_queries(self, math_format):
        """
        Produces queries from the topic, queries a WebMIaS index, and returns the queries along with
        the XML responses, and query results.

        Parameters
        ----------
        math_format : MathFormat
            The format in which the mathematical formulae will be represented in a query.

        Yields
        ------
        Query
            An unfinalized query along with the XML responses, and query results.
        """
        assert isinstance(math_format, MathFormat)

        for query in Query.from_topic(self, math_format):
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

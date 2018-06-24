"""
This module contains unit tests for the topic module.
"""

import unittest
from logging import getLogger

from lxml import etree

from .topic import Formula
from .util import xml_documents_equal


LOGGER = getLogger(__name__)


class TestFormula(unittest.TestCase):
    def setUp(self):
        self.formulae = [Formula.from_element(etree.fromstring(formula)) for formula in [
            r"""
                <formula id="f1.0" xmlns="http://ntcir-math.nii.ac.jp/"
                         xmlns:m="http://www.w3.org/1998/Math/MathML">
                  <m:math>
                    <m:semantics xml:id="m1.1a" xref="m1.1.pmml">
                      <m:apply xml:id="m1.1.8" xref="m1.1.8.pmml">
                        <m:eq xml:id="m1.1.5" xref="m1.1.5.pmml"/>
                        <m:apply xml:id="m1.1.8.1" xref="m1.1.8.1.pmml">
                          <m:times xml:id="m1.1.8.1.1" xref="m1.1.8.1.1.pmml"/>
                          <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="square"/>
                          <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="phi"/>
                        </m:apply>
                        <m:apply xml:id="m1.1.8.2" xref="m1.1.8.2.pmml">
                          <m:times xml:id="m1.1.8.2.1" xref="m1.1.8.2.1.pmml"/>
                          <m:ci xml:id="m1.1.6" xref="m1.1.6.pmml">i</m:ci>
                          <m:ci xml:id="m1.1.7" xref="m1.1.7.pmml">d</m:ci>
                        </m:apply>
                      </m:apply>
                      <m:annotation-xml encoding="MathML-Presentation" xml:id="m1.1.pmml"
                                        xref="m1.1">
                        <m:mrow xml:id="m1.1.8.pmml" xref="m1.1.8">
                          <m:mrow xml:id="m1.1.8.1.pmml" xref="m1.1.8.1">
                            <mws:qvar xmlns:mws="http://search.mathweb.org/ns"
                                      name="𝑠𝑞𝑢𝑎𝑟𝑒"/>
                            <m:mo xml:id="m1.1.8.1.1.pmml" xref="m1.1.8.1.1">⁢</m:mo>
                            <m:mrow xml:id="m1.1.3.pmml" xref="m1.1.3">
                              <m:mo xml:id="m1.1.3a.pmml" xref="m1.1.3">(</m:mo>
                              <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="𝑝ℎ𝑖"/>
                              <m:mo xml:id="m1.1.3c.pmml" xref="m1.1.3">)</m:mo>
                            </m:mrow>
                          </m:mrow>
                          <m:mo xml:id="m1.1.5.pmml" xref="m1.1.5">=</m:mo>
                          <m:mrow xml:id="m1.1.8.2.pmml" xref="m1.1.8.2">
                            <m:mi xml:id="m1.1.6.pmml" xref="m1.1.6">i</m:mi>
                            <m:mo xml:id="m1.1.8.2.1.pmml" xref="m1.1.8.2.1">⁢</m:mo>
                            <m:mi xml:id="m1.1.7.pmml" xref="m1.1.7">d</m:mi>
                          </m:mrow>
                        </m:mrow>
                      </m:annotation-xml>
                      <m:annotation encoding="application/x-tex" xml:id="m1.1b"
                                    xref="m1.1.pmml">\qvar{square}(\qvar{phi})=id</m:annotation>
                    </m:semantics>
                  </m:math>
                </formula>
            """,
            r"""
                <formula id="f1.1" xmlns="http://ntcir-math.nii.ac.jp/"
                         xmlns:m="http://www.w3.org/1998/Math/MathML">
                  <m:math>
                    <m:semantics xml:id="m2.1a" xref="m2.1.pmml">
                      <m:apply xml:id="m2.1.5" xref="m2.1.5.pmml">
                        <m:neq xml:id="m2.1.2" xref="m2.1.2.pmml"/>
                        <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="phi"/>
                        <m:apply xml:id="m2.1.5.1" xref="m2.1.5.1.pmml">
                          <m:times xml:id="m2.1.5.1.1" xref="m2.1.5.1.1.pmml"/>
                          <m:ci xml:id="m2.1.3" xref="m2.1.3.pmml">i</m:ci>
                          <m:ci xml:id="m2.1.4" xref="m2.1.4.pmml">d</m:ci>
                        </m:apply>
                      </m:apply>
                      <m:annotation-xml encoding="MathML-Presentation" xml:id="m2.1.pmml"
                                        xref="m2.1">
                        <m:mrow xml:id="m2.1.5.pmml" xref="m2.1.5">
                          <mws:qvar xmlns:mws="http://search.mathweb.org/ns" name="𝑝ℎ𝑖"/>
                          <m:mo xml:id="m2.1.2.pmml" xref="m2.1.2">≠</m:mo>
                          <m:mrow xml:id="m2.1.5.1.pmml" xref="m2.1.5.1">
                            <m:mi xml:id="m2.1.3.pmml" xref="m2.1.3">i</m:mi>
                            <m:mo xml:id="m2.1.5.1.1.pmml" xref="m2.1.5.1.1">⁢</m:mo>
                            <m:mi xml:id="m2.1.4.pmml" xref="m2.1.4">d</m:mi>
                          </m:mrow>
                        </m:mrow>
                      </m:annotation-xml>
                      <m:annotation encoding="application/x-tex" xml:id="m2.1b"
                                    xref="m2.1.pmml">\qvar{phi}\neq id</m:annotation>
                    </m:semantics>
                  </m:math>
                </formula>
            """,
        ]]

    def test_tex_text(self):
        self.assertEqual(self.formulae[0].tex_text, r"$ s ( p )=id$")
        self.assertEqual(self.formulae[1].tex_text, r"$ p \neq id$")

    def test_pmath_text(self):
        self.assertTrue(xml_documents_equal(
            self.formulae[0].pmath_text, r"""
                <math>
                  <mrow xml:id="m1.1.8.pmml" xref="m1.1.8">
                    <mrow xml:id="m1.1.8.1.pmml" xref="m1.1.8.1">
                      <mi>&#119904;</mi><mo xml:id="m1.1.8.1.1.pmml" xref="m1.1.8.1.1">&#8290;</mo>
                      <mrow xml:id="m1.1.3.pmml" xref="m1.1.3">
                        <mo xml:id="m1.1.3a.pmml" xref="m1.1.3">(</mo>
                        <mi>&#119901;</mi><mo xml:id="m1.1.3c.pmml" xref="m1.1.3">)</mo>
                      </mrow>
                    </mrow>
                    <mo xml:id="m1.1.5.pmml" xref="m1.1.5">=</mo>
                    <mrow xml:id="m1.1.8.2.pmml" xref="m1.1.8.2">
                      <mi xml:id="m1.1.6.pmml" xref="m1.1.6">i</mi>
                      <mo xml:id="m1.1.8.2.1.pmml" xref="m1.1.8.2.1">&#8290;</mo>
                      <mi xml:id="m1.1.7.pmml" xref="m1.1.7">d</mi>
                    </mrow>
                  </mrow>
                </math>
            """))
        self.assertTrue(xml_documents_equal(
            self.formulae[1].pmath_text, r"""
                <math>
                    <mrow xml:id="m2.1.5.pmml" xref="m2.1.5">
                        <mi>&#119901;</mi><mo xml:id="m2.1.2.pmml" xref="m2.1.2">&#8800;</mo>
                        <mrow xml:id="m2.1.5.1.pmml" xref="m2.1.5.1">
                            <mi xml:id="m2.1.3.pmml" xref="m2.1.3">i</mi>
                            <mo xml:id="m2.1.5.1.1.pmml" xref="m2.1.5.1.1">&#8290;</mo>
                            <mi xml:id="m2.1.4.pmml" xref="m2.1.4">d</mi>
                        </mrow>
                    </mrow>
                </math>
            """))

    def test_cmath_text(self):
        self.assertTrue(xml_documents_equal(
            self.formulae[0].cmath_text, r"""
                <math>
                  <apply xml:id="m1.1.8" xref="m1.1.8.pmml">
                    <eq xml:id="m1.1.5" xref="m1.1.5.pmml"/>
                    <apply xml:id="m1.1.8.1" xref="m1.1.8.1.pmml">
                      <times xml:id="m1.1.8.1.1" xref="m1.1.8.1.1.pmml"/>
                      <ci>s</ci><ci>p</ci>
                    </apply>
                    <apply xml:id="m1.1.8.2" xref="m1.1.8.2.pmml">
                      <times xml:id="m1.1.8.2.1" xref="m1.1.8.2.1.pmml"/>
                      <ci xml:id="m1.1.6" xref="m1.1.6.pmml">i</ci>
                      <ci xml:id="m1.1.7" xref="m1.1.7.pmml">d</ci>
                    </apply>
                  </apply>
                </math>
            """))
        self.assertTrue(xml_documents_equal(
            self.formulae[1].cmath_text, r"""
                <math>
                    <apply xml:id="m2.1.5" xref="m2.1.5.pmml">
                        <neq xml:id="m2.1.2" xref="m2.1.2.pmml"/>
                        <ci>p</ci>
                        <apply xml:id="m2.1.5.1" xref="m2.1.5.1.pmml">
                            <times xml:id="m2.1.5.1.1" xref="m2.1.5.1.1.pmml"/>
                            <ci xml:id="m2.1.3" xref="m2.1.3.pmml">i</ci>
                            <ci xml:id="m2.1.4" xref="m2.1.4.pmml">d</ci>
                        </apply>
                    </apply>
                </math>
            """))


if __name__ == '__main__':
    unittest.main()

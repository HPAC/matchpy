---
title: 'MatchPy: Pattern Matching in Python'
tags:
  - pattern matching
  - term rewriting
  - many-to-one matching
authors:
 - name: Manuel Krebber
   orcid: 0000-0001-5038-1102
   affiliation: 1
 - name: Henrik Barthels
   orcid: 0000-0001-6744-3605
   affiliation: 1
affiliations:
 - name: AICES, RWTH Aachen University
   index: 1
date: 22 January 2018
bibliography: paper.bib
---

# Summary

Pattern matching is a powerful tool for symbolic computations. Applications include symbolic integration, term rewriting systems, theorem proving and the manipulation of abstract syntax trees. Given a pattern and an expression, the goal of pattern matching is to find a substitution for all the variables in the pattern such that the pattern becomes the expression. As an example, consider the pattern `f(x)`, where `x` is a variable, and the expression `f(a)`. Then the substitution that replaces `x` with `a` is a match. In practice, functions can also be associative and/or commutative, which makes matching more complicated and can lead to multiple possible matches. 

Among existing systems, Mathematica [@mathematica] arguably offers the most expressive pattern matching. Unfortunately, no lightweight implementation of pattern matching as general and flexible as Mathematica exists for Python. The purpose of MatchPy [@matchpy; @krebber2017:3] is to provide this functionality in Python. While the pattern matching in SymPy [@sympy] can work with associative/commutative functions, it does not support finding multiple matches, which is relevant in some applications. Furthermore, SymPy does not support sequence variables and is limited to a predefined set of mathematical operations.

## Many-to-One Matching

In many applications, a fixed set of patterns is matched repeatedly against different subjects. The simultaneous matching of multiple patterns is called many-to-one matching, as opposed to one-to-one matching which denotes matching with a single pattern. Many-to-one matching can achieve a significant speed increase compared to one-to-one matching by exploiting similarities between patterns. MatchPy includes efficient algorithms for many-to-one matching [@krebber2017:2], as opposed to Mathematica and SymPy.

The basic algorithms implemented in MatchPy have been described in a Master thesis [@krebber2017:1].

## Use in Ongoing Research

MatchPy is a central part of [@linnea], an experimental tool for the automatic generation of optimized code for linear algebra problems.


# References

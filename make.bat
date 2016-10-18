@ECHO off
if /I %1 == init goto :init
if /I %1 == test goto :test
if /I %1 == check goto :check
if /I %1 == coverage goto :coverage

goto :eof

:init
	pip install -r requirements.txt
goto :eof

:test
	python -m unittest discover tests
goto :eof

:check
	pylint patternmatcher
goto :eof

:coverage
	coverage run --source patternmatcher/syntactic.py -m tests.test_syntactic
	coverage run -a --source patternmatcher/functions.py -m tests.test_functions
	coverage run -a --source patternmatcher/utils.py -m tests.test_utils
	coverage run -a --source patternmatcher/bipartite.py -m tests.test_bipartite
	coverage run -a --source patternmatcher/matching.py -m tests.test_matching
	coverage run -a --source patternmatcher/expressions.py -m tests.test_expressions
	coverage html
goto :eof

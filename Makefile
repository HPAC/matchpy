init:
	pip install -r requirements.txt

test:
	python -m unittest discover tests

check:
	pylint patternmatcher

coverage:
	coverage run --source patternmatcher/syntactic.py -m tests.test_syntactic
	coverage run -a --source patternmatcher/functions.py -m tests.test_functions
	coverage run -a --source patternmatcher/utils.py -m tests.test_utils
	coverage run -a --source patternmatcher/bipartite.py -m tests.test_bipartite
	coverage run -a --source patternmatcher/matching.py -m tests.test_matching
	coverage run -a --source patternmatcher/multiset.py -m tests.test_multiset
	coverage run -a --source patternmatcher/expressions.py -m tests.test_expressions
	coverage html

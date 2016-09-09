init:
	pip install -r requirements.txt

test:
	python -m unittest discover tests

check:
	pylint patternmatcher

coverage:
	coverage run --branch --source patternmatcher/syntactic.py -m tests.test_syntactic
	coverage run -a --branch --source patternmatcher/functions.py -m tests.test_functions
	coverage run -a --branch --source patternmatcher/utils.py -m tests.test_utils
	coverage run -a --branch --source patternmatcher/bipartite.py -m tests.test_bipartite
	coverage run -a --branch --source patternmatcher/matching.py -m tests.test_matching
	coverage html

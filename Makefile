init:
	pip install -r requirements.txt

test:
	python -m unittest discover tests

check:
	pylint patternmatcher

coverage:
    coverage run -m --branch tests.test_syntactic
    coverage html

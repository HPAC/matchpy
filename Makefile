init:
	pip install -r requirements.txt

test:
	py.test tests/ --doctest-modules patternmatcher/

check:
	pylint patternmatcher

coverage:
	py.test --cov=patternmatcher --cov-report html --cov-report term tests/

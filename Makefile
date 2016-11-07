init:
	pip install -r requirements.txt

test:
	pytest tests

check:
	pylint patternmatcher

coverage:
	pytest --cov=patternmatcher --cov-report html --cov-report term tests/

init:
	pip install -r requirements.txt

test:
	py.test tests/ --doctest-modules patternmatcher/ README.rst

doctest:
	py.test --doctest-modules -k "not tests" patternmatcher/ README.rst

check:
	pylint patternmatcher

coverage:
	py.test --cov=patternmatcher --cov-report html --cov-report term tests/

api-docs:
	rmdir docs/api
	sphinx-apidoc -n -e -T -o docs/api patternmatcher
	make docs

docs:
	cd docs/api
	make html
	cd ..

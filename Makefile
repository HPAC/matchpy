init:
	pip install -r requirements.txt

test:
	py.test tests/ --doctest-modules matchpy/ README.rst docs/example.rst

doctest:
	py.test --doctest-modules -k "not tests" matchpy/ README.rst docs/example.rst

check:
	pylint matchpy

coverage:
	py.test --cov=matchpy --cov-report html --cov-report term tests/

api-docs:
	rmdir docs/api
	sphinx-apidoc -n -e -T -o docs/api matchpy
	make docs

docs:
	cd docs
	make html
	cd ..

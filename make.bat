@ECHO off
if /I %1 == init goto :init
if /I %1 == test goto :test
if /I %1 == doctest goto :doctest
if /I %1 == check goto :check
if /I %1 == coverage goto :coverage
if /I %1 == api-docs goto :apidocs
if /I %1 == docs goto :docs

goto :eof

:init
	pip install -r requirements.txt
goto :eof

:test
	py.test tests\ --doctest-modules matchpy\ README.rst docs\example.rst
goto :eof

:doctest
	py.test --doctest-modules -k "not tests" matchpy\ README.rst docs\example.rst
goto :eof

:check
	pylint --reports=no matchpy
goto :eof

:coverage
	py.test --cov=matchpy --cov-report html --cov-report term tests\
goto :eof

:apidocs
	rmdir /s /q docs\api
	sphinx-apidoc -e -T -o docs\api matchpy
goto :docs

:docs
	cd docs
	make html
	cd ..
goto :eof


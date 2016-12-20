@ECHO off
if /I %1 == init goto :init
if /I %1 == test goto :test
if /I %1 == check goto :check
if /I %1 == coverage goto :coverage
if /I %1 == api-docs goto :apidocs
if /I %1 == docs goto :docs

goto :eof

:init
	pip install -r requirements.txt
goto :eof

:test
	py.test tests\ --doctest-modules patternmatcher\ README.rst
goto :eof

:check
	pylint patternmatcher
goto :eof

:coverage
	py.test --cov=patternmatcher --cov-report html --cov-report term tests\
goto :eof

:apidocs
	rmdir /s /q docs\api
	sphinx-apidoc -e -T -o docs\api patternmatcher
goto :docs

:docs
	cd docs
	make html
	cd ..
goto :eof


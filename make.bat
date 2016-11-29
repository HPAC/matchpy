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
	py.test tests\ --doctest-modules patternmatcher\ README.rst
goto :eof

:check
	pylint patternmatcher
goto :eof

:coverage
	py.test --cov=patternmatcher --cov-report html --cov-report term tests\
goto :eof

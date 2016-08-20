@ECHO off    
if /I %1 == init goto :init
if /I %1 == test goto :test
if /I %1 == check goto :check
if /I %1 == coverage goto :coverage

goto :eof

:init
	pip install -r requirements.txt

:test
	python -m unittest discover tests

:check
	pylint patternmatcher

:coverage
    coverage run -m --branch tests.test_syntactic
    coverage run -a -m --branch tests.test_functions
    coverage run -a -m --branch tests.test_utils
    coverage html

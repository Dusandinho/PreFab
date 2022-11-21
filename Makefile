
install:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .
	pre-commit install

test:
	pytest

cov:
	pytest --cov=fabmodel

mypy:
	mypy . --ignore-missing-imports

flake8:
	flake8 --select RST

pylint:
	pylint fabmodel

pydocstyle:
	pydocstyle fabmodel

doc8:
	doc8 docs/

update:
	pur

update-pre:
	pre-commit autoupdate --bleeding-edge

release:
	git push
	git push origin --tags

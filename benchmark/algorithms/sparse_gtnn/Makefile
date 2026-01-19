PYTHON ?= python3
PIP ?= pip

.PHONY: all clean build test pyi

all: clean build pyi

clean:
	rm -rf build *.so *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

build:
	$(PIP) install --force-reinstall --no-cache-dir -v .

pyi:
	stubgen -m sparse_gtnn -o .

test:
	$(PYTHON) test.py

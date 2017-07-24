fetch:
	python3 lib/download_stocks.py

run:
	python3 main.py

test:
	python3 -m unittest discover -s tests/ -p "*_test.py"

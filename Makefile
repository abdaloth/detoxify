locust: # Use with make locust host=<host>
	locust --host $(host) --locustfile locustfile.py

install:
	pip install --upgrade pip
	pip install -r requirements.txt

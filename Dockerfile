# This image is for use as a custom runtime with Google App Engine
# See: https://cloud.google.com/appengine/docs/flexible/custom-runtimes/build
FROM gcr.io/google-appengine/python

# Flask App
ADD ./main_local.py main.py
ADD requirements.txt requirements.txt
ADD ./templates templates

# Serialized model and tokenizer
ADD ./serialized serialized

# Install dependencides
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python3", "main.py"]
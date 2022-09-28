From tensorflow/tensorflow:2.8.3-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev wget vim python3.7

ADD ../tfts/ /tfts/
ADD ../examples /examples/
COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
WORKDIR examples

ENTRYPOINT ["python"]
CMD ["run_rnn.py"]

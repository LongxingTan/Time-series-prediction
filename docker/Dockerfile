From tensorflow/tensorflow:2.16.1-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev wget vim python3.9

RUN pip install --no-cache-dir tfts

# Set the default command to python3.
CMD ["python3"]

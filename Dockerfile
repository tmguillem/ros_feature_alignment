FROM duckietown/rpi-duckiebot-base:master18

RUN [ "cross-build-start" ]

RUN apt-get update && \
    pip install -r requirements.txt

ENV READTHEDOCS True

RUN mkdir /home/software/catkin_ws/src/feature_alignment
COPY . /home/software/catkin_ws/src/feature_alignment/

RUN [ "cross-build-end" ]

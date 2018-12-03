FROM duckietown/rpi-duckiebot-base:master18

RUN [ "cross-build-start" ]

RUN apt-get update && \
    pip install -r requirements.txt

RUN [ "cross-build-end" ]

COPY Images/. /feature_matching/Images/.
COPY python_scripts/. /feature_matching/python_scripts/.
FROM python:3.9-slim

# setup home directory of the container
WORKDIR /home/app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model
COPY model model

# copy code
COPY docker wsd
COPY code/src/model.py wsd/src/wsd_model.py
ENV PYTHONPATH wsd

# standard cmd
CMD [ "python", "wsd/app.py" ]

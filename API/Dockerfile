# API dockerfile

FROM python:3.10

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

# set a directory for the api
WORKDIR /api

# Adding requirements file to current directory
# just this file first to cache the pip install step when code changes
COPY requirements.txt .

# print debug information
RUN cat requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy all the files to the container
COPY . .

# define the port number the container should expose
EXPOSE 8000 

# run the command
CMD python -m uvicorn ml_api:api --host 0.0.0.0 --port 8000
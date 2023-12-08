# app Dockerfile
FROM python:3.10

# run this before copying requirements for cache efficiency
RUN pip install --upgrade pip

# set a directory for the app
WORKDIR /app

# Adding requirements file to current directory
# just this file first to cache the pip install step when code changes
COPY requirements.txt .

# print debug information
RUN cat requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy all the files to the container
COPY . .

# copy input data only on GitHub
RUN wget https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip
RUN unzip Projet+Mise+en+prod+-+home-credit-default-risk

# define the port number the container should expose
EXPOSE 8501 

# run the command
CMD ["streamlit", "run", "./app.py"]

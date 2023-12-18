This is my first ever coding project, using GitHub.

It is called credit_scoring.
It is part of a Data Science Masters program I am enrolled in.

The objective of the project is to get a sense of what a full Data Science project is, from data cleaning and modelisation to app/API development and deployment.

As part of this project, I created a model to predict an applicant's payment default risk level based on various candidate information. This is to inform a decision to grant or not a new credit to the applicant.
The model was trained, versions were stored through MLFlow and the best version was deployed.

A Streamlit app was implemented to provide account managers with information regarding an applicant, scoring and credit granting decision, as well as detailed information supporting the decision.
An API was implemented using FastAPI to query the model and get corresponding predictions and prediction explanation.

The code was pushed to GitHub a monorepo for simplicity but organized in 2 separate folders (app and API) as a way to balance for autonomy of both parts of the application.

A couple of unit tests were added to the API using pytest, and a GitHub Actions build worklow was created to automatically conduct corresponding jobs/steps as well as independent requirements installations for the webapp and the API.

Both parts of the application were then independently containerized through Docker (and docker-compose so they could communicate) and deployed to an AWS EC2 instance. Corresponding jobs/steps were added to the GitHub Actions build and deployment workflow along with an AWS authentication mechanism so as to obtain a full automated deployment pipeline.
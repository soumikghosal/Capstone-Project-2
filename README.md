# Capstone Project 2
### Business Problem

Predicting Diabetes to Improve Preventive Healthcare and Patient Outcomes: In many healthcare systems, identifying individuals at risk for diabetes early can significantly improve outcomes by enabling timely intervention. Diabetes is a chronic disease with substantial healthcare costs and long-term complications if left unmanaged. 

##### Problem Statement 
How can we use data to predict whether a person is likely to have diabetes, enabling healthcare providers to reduce disease progression and associated costs?

### Use of the Machine Learning Model
##### Benefits
1. Early Detection:
Identifying at-risk individuals before they develop severe symptoms can lead to better health outcomes.

2. Cost Reduction:
By preventing complications, the healthcare system can save costs associated with advanced diabetes treatments (e.g., dialysis, surgeries).

3. Personalized Care:
Patients can receive tailored advice and interventions based on their risk profile.

### About the dataset
The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.

## FOLDER STRUCTURE
- codes: This folder contains the python files
    - train.py: This file reads the data and builds multiple models and chooses the best model (Logistic Regression) based on the performance. Please refer to the Modelling.ipynb in the notebook folder, how we select this LR.
    - predict.py: Flask app
    - test.py: Used to test the Flask app
    - lambda_function.py
    - test_lambda_local.py
    - test_lambda_main.py 

- data: This folder contains the data file used for the project.

- models: This folder contains the pickled models saved.

- notebooks: This folder contains the jupyter notebook(s) created. It contains the EDA and modelling steps taken to select the best model. 


## Set up Environment

After cloning the [repository](https://github.com/soumikghosal/Capstone-Project-2), follow the below steps:

Open cmd and navigate to the cloned local repository folder and create a virtual environment with the desired environment name.
```
virtualenv capstone_2
```
Activate the environment
```
capstone_2\Scripts\activate
```
Installl the required python packages
```
pip install -r requirement.txt
```

# Running and Testing Service Locally with Flask

*** Note: Please rename Dockerfile as Dockerfile_1 and Dockerfile_2 as Dockerfile (in this order)

Build the docker container.
```
docker build -t capstone_test_2 .
```
Run the docker container
```
docker run -it --rm -p 9696:9696 capstone_test_2:latest
```
Testing the service
```
python scripts\test.py
```
### API Request format

Below is an example of the data to  be fed to the API. Please refer to the attribute information to find the relevant and valid values for each variables.
```
client = {"gender": "Male",
          "age": 33.0,
          "hypertension": 0,
          "heart_disease": 0,
          "smoking_history": "former",
          "bmi": 35.81,
          "hba1c_level": 5.8,
          "blood_glucose_level": 140
        }
```

## Running and Testing AWS Lambda Remotely

*** Note: If you ran the previous section then please rename Dockerfile as Dockerfile_2 and Dockerfile_1 as Dockerfile (in this order) or else run following steps.

Build the docker container.
```
docker build -t capstone_test_2 .
```
Run the docker container
```
docker run -it --rm -p 8080:8080 capstone_test_2:latest
```
Testing the service locally at: http://localhost:8080/2015-03-31/functions/function/invocations
```
python scripts\test_lambda_local.py
```

If you dont have an AWS account please go through this [video](https://www.youtube.com/watch?v=kBch5oD5BkY&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=87). It will cover:
- Create a repository and login to Docker
- Publishing the image to AWS ECR
- Creating AWS Lambda function
- Testing the function from the AWS Console

** Note: You will need to configure AWS CLI

Once this is done follow this [video](https://www.youtube.com/watch?v=wyZ9aqQOXvs&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=88). It covers how to expose the Lambda using API Gateway


Testing the service remotely: I have already hosted this service via AWS Lambda function at: https://exkg23rhe2.execute-api.us-east-2.amazonaws.com/capstone-1/predict
Feel free to test it by running the below command. The url is already set in the test_lambda_main.py file. If you wish to test your app, replace the url in this file and run the same below command.

```
python scripts\test_lambda_main.py
```


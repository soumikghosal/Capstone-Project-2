FROM public.ecr.aws/lambda/python:3.12-x86_64

COPY ["requirement.txt", "./"]
COPY ["./scripts/lambda_function.py", "./"]
COPY ["./models/diabetes_pred_model.bin", "./"]

RUN pip install -r requirement.txt

CMD ["lambda_function.lambda_handler"]

#Choose base image
FROM python:3.9

#Define work directory inside the container
WORKDIR /app

#Copy the API files to the container
COPY requirements.txt .
COPY app.py .
COPY final_churn_prediction.pkl .

#Install the necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Expose the port that Flask will use
EXPOSE 8080

#Define the command to start Flask API
CMD ["python", "app.py"]
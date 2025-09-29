# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Define the command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
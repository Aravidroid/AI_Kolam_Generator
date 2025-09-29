FROM python:3.11-slim
# Set the working directory to the sub-folder
WORKDIR /app/kolam_ai_app
# Copy requirements from the sub-folder
COPY kolam_ai_app/requirements.txt .
# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of your application code from the sub-folder
COPY kolam_ai_app/ .
# Define the command to run your app
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]

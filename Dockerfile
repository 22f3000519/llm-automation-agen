# Start from a base image that supports Python
FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean

# Copy and install dependencies first
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Set the path for the git executable
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git



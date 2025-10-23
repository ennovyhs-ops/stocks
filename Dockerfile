# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y libstdc++6

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run app.py when the container launches
CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]
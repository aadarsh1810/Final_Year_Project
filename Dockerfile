# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the BERT model file into the container (make sure bert_model_10epochs.pth is in the same folder as Dockerfile)
COPY bert_model_10epochs.pth /app/

# Copy the rest of the application into the container
COPY . .

# Expose port for the Streamlit app (default: 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]

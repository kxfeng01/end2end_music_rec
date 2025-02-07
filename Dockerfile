
# Use an official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies (curl, tar, and Java for Spark)
RUN apt-get update && apt-get install -y openjdk-17-jre curl tar


# Set Java and Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark-3.3.0-bin-hadoop3
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Download and extract Spark
RUN curl -fsSL https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz \
    | tar xz -C /opt

# Copy application files
COPY streamlit_app.py .
COPY local_export/ALSModel_Standalone /app/local_export/ALSModel_Standalone

# Expose the FastAPI port
EXPOSE 8000

# Run app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]

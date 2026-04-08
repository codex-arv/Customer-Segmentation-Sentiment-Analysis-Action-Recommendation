FROM python:3.10-slim

# system dependencies for ML libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# expose container port
EXPOSE 10000

# start application
CMD ["python", "run.py"]
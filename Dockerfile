FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install supervisor for managing multiple processes
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Copy all project files
COPY . .

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8501

CMD ["/usr/bin/supervisord"]

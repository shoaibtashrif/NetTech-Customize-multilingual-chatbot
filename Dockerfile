FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy all files to the working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

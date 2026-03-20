FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train the model
RUN python -m src.pipeline.train_pipeline

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "application.py"]

# 1. Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy the dependencies list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the backend application code
COPY . .

# 5. Expose port 8000 for external communication
EXPOSE 8000

# 6. Command to start the FastAPI server when the container runs
# (Assuming your main application file is named main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]   
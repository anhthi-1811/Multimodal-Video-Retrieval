# 1. Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the frontend application code
COPY . .

# 5. Expose port 8501 (Streamlit's default port)
EXPOSE 8501

# 6. Command to run the Streamlit web application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first, install dependencies — done as a separate step
# before copying the rest of the code so Docker can cache this layer.
# If only your .py files change later, Docker won't need to reinstall
# every package from scratch, just reuse this cached layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the saved model artifacts
COPY src/ ./src/
COPY data/*.joblib ./data/

EXPOSE 5000

CMD ["python", "src/app.py"]
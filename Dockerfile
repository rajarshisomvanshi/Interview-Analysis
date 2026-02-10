FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MinerU/Magic-PDF dependencies if needed for full features, 
# but for now we stick to the requirements.txt which has pymupdf
# If MinerU is strictly required, we'd need more system deps.
# Based on plan, we use Dextora OCR (API) + PyMuPDF fallback, so slim is fine.

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "2000"]

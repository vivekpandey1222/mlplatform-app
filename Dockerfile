# Stage 1: डिपेंडेंसी इंस्टॉल करना (यह स्टेज बाद में हटा दी जाएगी)
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: अंतिम छोटी और सुरक्षित इमेज
FROM python:3.12-slim
RUN useradd --uid 10001 --no-create-home appuser
WORKDIR /srv
COPY --from=builder /install /usr/local
COPY app ./app
COPY model ./model
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


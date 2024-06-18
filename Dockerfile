# Gunakan image Python versi 3.12.2
FROM python:3.9

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Upgrade PIP
RUN pip install --upgrade pip

# Install dependencies menggunakan pip
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi Flask ke dalam container
COPY . .

# Jalankan aplikasi Flask ketika container dijalankan
CMD ["python", "deploy.py"]
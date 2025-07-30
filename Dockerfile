FROM python:3.10-slim

WORKDIR /app

# 先升级 pip
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

# 安装依赖，增加超时时间
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

COPY . .

RUN touch property_translations.db

EXPOSE 8080

CMD ["python", "main.py"]

FROM python:3.11-slim

WORKDIR /app

# 1. 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 拷贝代码（不包含 entrypoint.sh）
COPY . .

# 3. 再单独拷贝 entrypoint + chmod（关键）
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 5250

ENTRYPOINT ["./entrypoint.sh"]

# 使用官方的 Python 3.9 slim 版本作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到工作目录
COPY requirements.txt requirements.txt

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将项目代码复制到工作目录
COPY . .

# 确保数据库文件存在（如果不存在则创建空文件）
RUN touch property_translations.db

# 容器对外暴露的端口
EXPOSE 8080

# 容器启动时运行的命令
CMD ["python", "main.py"]

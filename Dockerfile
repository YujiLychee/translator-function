# --- 基礎鏡像 ---
# 使用官方的 Python 3.9 的輕量級版本作為基礎
FROM python:3.9-slim

# --- 設置工作環境 ---
# 設置容器內的工作目錄為 /app
WORKDIR /app

# --- 安裝依賴 ---
# 首先只複製依賴文件，這樣可以利用 Docker 的緩存機制
# 只有當 requirements.txt 改變時，才會重新執行安裝步驟
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 複製應用程序代碼和數據 ---
# 將當前文件夾下的所有文件（包括 main.py, translate.py, 和 property_translations.db）複製到容器的 /app 目錄中
COPY . .

# --- 設置環境變量 ---
# 設置服務器監聽的端口。Cloud Run 會自動將外部請求映射到這個端口
ENV PORT 8080

# --- 容器啟動命令 ---
# 使用 functions-framework 啟動服務
# --target=translate_handler 指定要運行的函數名稱，對應您 main.py 中的函數
CMD ["functions-framework", "--target=translate_handler", "--port=8080"]
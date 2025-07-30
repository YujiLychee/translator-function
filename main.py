from flask import jsonify
from dataclasses import asdict
import logging

# 導入您的核心翻譯系統
from translate import PropertyTranslationSystem

# --- 全局初始化 ---
logging.basicConfig(level=logging.INFO)
try:
    logging.info("Translator API: 正在進行冷啟動初始化...")
    # 注意：這裡我們仍然需要 db 文件來初始化
    # 您可以通過環境變量傳遞 GROK_API_KEY，這更安全
    import os
    grok_api_key = os.environ.get("GROK_API_KEY", "YOUR_DEFAULT_KEY_HERE")
    translator = PropertyTranslationSystem(
        db_path="property_translations.db",
        grok_api_key=grok_api_key
    )
    logging.info("✅ Translator API: 初始化完成。")
except Exception as e:
    logging.error(f"❌ Translator API: 初始化時發生嚴重錯誤: {e}")
    translator = None

# --- Google Cloud Function 入口函數 ---
def translate_handler(request):
    # 處理 CORS
    if request.method == 'OPTIONS':
        headers = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Max-Age': '3600'}
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    if not translator:
        return (jsonify({"error": "Translator 未能成功初始化。"}), 500, headers)

    request_json = request.get_json(silent=True)
    if not request_json or 'name' not in request_json:
        return (jsonify({"error": "請求體必須是包含 'name' 鍵的 JSON。"}), 400, headers)

    try:
        translation_result = translator.translate(
            property_name=request_json['name'],
            context=request_json.get('context', None)
        )
        result_dict = asdict(translation_result)
        return (jsonify(result_dict), 200, headers)
    except Exception as e:
        logging.error(f"翻譯過程中發生錯誤: {e}", exc_info=True)
        return (jsonify({"error": "服務器內部翻譯錯誤。"}), 500, headers)

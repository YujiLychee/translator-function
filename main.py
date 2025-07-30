from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import os

# 导入您的核心翻译系统
from translate import PropertyTranslationSystem

# 创建 Flask 应用
app = Flask(__name__)

# --- 全局初始化 ---
logging.basicConfig(level=logging.INFO)
translator = None

def init_translator():
    global translator
    try:
        logging.info("Translator API: 正在进行冷启动初始化...")
        # 从环境变量获取 API KEY
        grok_api_key = os.environ.get("GROK_API_KEY", "YOUR_DEFAULT_KEY_HERE")
        
        # 检查数据库文件是否存在
        db_path = "property_translations.db"
        if not os.path.exists(db_path):
            logging.warning(f"数据库文件 {db_path} 不存在，创建空文件")
            # 创建一个空文件，让 PropertyTranslationDatabase 来初始化
            open(db_path, 'a').close()
        
        translator = PropertyTranslationSystem(
            db_path=db_path,
            grok_api_key=grok_api_key
        )
        logging.info(" Translator API: 初始化完成。")
    except Exception as e:
        logging.error(f" Translator API: 初始化时发生严重错误: {e}")
        translator = None

# 健康检查路由
@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "translator-api"}), 200

# 主要的翻译路由
@app.route('/translate', methods=['POST', 'OPTIONS'])
def translate_handler():
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    if not translator:
        init_translator()  # 重试初始化
        
    if not translator:
        return jsonify({"error": "Translator 未能成功初始化。"}), 500

    request_json = request.get_json(silent=True)
    if not request_json or 'name' not in request_json:
        return jsonify({"error": "请求体必须是包含 'name' 键的 JSON。"}), 400

    try:
        translation_result = translator.translate(
            property_name=request_json['name'],
            context=request_json.get('context', None)
        )
        result_dict = asdict(translation_result)
        return jsonify(result_dict), 200
    except Exception as e:
        logging.error(f"翻译过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部翻译错误。"}), 500

# 初始化翻译器
init_translator()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

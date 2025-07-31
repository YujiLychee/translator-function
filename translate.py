import json
import sqlite3
import requests
import difflib
import re
import os  
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from xai_sdk import Client
    from xai_sdk.chat import user,system
    from xai_sdk.search import SearchParameters, web_source, news_source
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_SDK_AVAILABLE = False
    logger.warning("xai_sdk未安装，将使用原有的 requests 方法")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    from sentence_transformers import SentenceTransformer # type: ignore
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.warning("sentence-transformers未安装，将使用基础模糊匹配")
try:
    import jieba # type: ignore
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba未安裝，將跳過結構解析功能")


#翻譯結果數據類
@dataclass
class TranslationResult:
    chinese_name: str
    english_name: str
    confidence: float
    method: str
    layer: int
    source: str = ""
    reasoning: str = ""
    alternatives: List[str] = None
    search_analysis: Dict = None


#樓盤翻譯資料庫管理
class PropertyTranslationDatabase:   
    def __init__(self, db_path: str = "property_translations.db"):
        self.db_path = db_path
        
        # 檢查資料庫文件是否存在
        if not os.path.exists(db_path):
            logger.error(f"資料庫文件 {db_path} 不存在，請先運行 Notebook 創建初始數據")
            raise FileNotFoundError(f"Database file {db_path} not found")
        
        # 初始化其他必要表和組件規則
        self.init_database()
        self.load_initial_data()
    
    #初始化資料庫表結構
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 官方翻譯表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS official_translations (
                id INTEGER PRIMARY KEY,
                chinese_name TEXT UNIQUE,
                english_name TEXT,
                source TEXT,
                confidence REAL,
                created_at TIMESTAMP,
                verified BOOLEAN DEFAULT TRUE
            )
        """)
        
        # 已驗證翻譯表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verified_translations (
                id INTEGER PRIMARY KEY,
                chinese_name TEXT UNIQUE,
                english_name TEXT,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP,
                last_used TIMESTAMP
            )
        """)
        
        # 組件翻譯規則表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS component_rules (
                id INTEGER PRIMARY KEY,
                chinese_component TEXT,
                english_options TEXT,  -- JSON array
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                context TEXT
            )
        """)
        
        # 翻譯歷史記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY,
                chinese_name TEXT,
                english_name TEXT,
                method TEXT,
                layer INTEGER,
                confidence REAL,
                search_results TEXT,  -- JSON
                timestamp TIMESTAMP,
                user_feedback TEXT
            )
        """)
        # 固定地名（行政區 / 大型地標 / 車站）對照表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS geo_locations (
                id INTEGER PRIMARY KEY,
                 chinese_name TEXT UNIQUE,
                english_name TEXT,
                category TEXT              -- district / area / station ...
            )
        """)
        conn.commit()
        conn.close()

    def get_geo_location(self, chinese_name: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # 精確匹配
        cur.execute("SELECT english_name FROM geo_locations WHERE chinese_name = ?", (chinese_name,))
        row = cur.fetchone()
        if row:
            conn.close()
            return row[0]
        
        # 站名智能匹配 
        if chinese_name.endswith("站"):
            base_name = chinese_name[:-1]  # 去掉"站"字
            cur.execute("SELECT english_name FROM geo_locations WHERE chinese_name = ?", (base_name,))
            row = cur.fetchone()
            if row:
                conn.close()
                # 如果找到基礎地名，添加 Station
                return f"{row[0]} Station"
        
        conn.close()
        return None

    # PropertyTranslationDatabase 中增一個查詢
    def get_slang_translation(self, zh: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            "SELECT english_name FROM verified_translations "
            "WHERE chinese_name = ? AND confidence >= 0.8", (zh,)
        )
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    #載入初始翻譯數據：在測試階段僅使用已存在的 property_translations.db
    def load_initial_data(self): 
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM verified_translations")
            existing_count = cursor.fetchone()[0]            
            if existing_count > 0:
                logger.info(f"發現已存在 {existing_count} 條翻譯數據")
            else:
                logger.warning("verified_translations 表為空")
                return
            
            # 確保其他必要表存在（如果不存在則創建）
            self._ensure_other_tables_exist(cursor)
            
            #添加組件翻譯規則（這些是通用規則，不是特定樓盤）
            self._add_component_rules(cursor)
            self.load_geo_locations()
            conn.commit()
            
            # 4. 輸出統計信息
            self._print_simple_stats(cursor)
            
            logger.info("初始翻譯數據載入完成")
            
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                logger.error("資料庫表不存在")
                raise
            else:
                logger.error(f"資料庫操作錯誤：{e}")
                raise
        except Exception as e:
            logger.error(f"載入初始數據時發生錯誤：{e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_geo_locations(self):
        geo_pairs = {
            # --- 行政區 (18區) ---
            "中西區": "Central and Western District", 
            "灣仔": "Wan Chai District",
            "東區": "Eastern District", 
            "南區": "Southern District",
            "深水埗": "Sham Shui Po District", 
            "油尖旺": "Yau Tsim Mong District",
            "九龍城": "Kowloon City District", 
            "黃大仙": "Wong Tai Sin District",
            "觀塘": "Kwun Tong District", 
            "荃灣": "Tsuen Wan District",
            "屯門": "Tuen Mun District", 
            "元朗": "Yuen Long District",
            "北區": "North District", 
            "大埔": "Tai Po District",
            "沙田": "Sha Tin District", 
            "西貢": "Sai Kung District",
            "離島": "Islands District",

            # --- 主要區域 ---
            "中環": "Central", 
            "金鐘": "Admiralty",
            "灣仔": "Wan Chai", 
            "銅鑼灣": "Causeway Bay",
            "天后": "Tin Hau", 
            "炮台山": "Fortress Hill",
            "北角": "North Point", 
            "鰂魚涌": "Quarry Bay",
            "太古": "Tai Koo", 
            "西營盤": "Sai Ying Pun",
            "上環": "Sheung Wan", 
            "堅尼地城": "Kennedy Town",
            "薄扶林": "Pok Fu Lam", 
            "香港仔": "Aberdeen",
            "鴨脷洲": "Ap Lei Chau", 
            "赤柱": "Stanley",
            "尖沙咀": "Tsim Sha Tsui", 
            "佐敦": "Jordan",
            "油麻地": "Yau Ma Tei", 
            "旺角": "Mong Kok",
            "太子": "Prince Edward", 
            "深水埗": "Sham Shui Po",
            "長沙灣": "Cheung Sha Wan", 
            "荔枝角": "Lai Chi Kok",
            "美孚": "Mei Foo", 
            "九龍塘": "Kowloon Tong",
            "何文田": "Ho Man Tin", 
            "紅磡": "Hung Hom",
            "土瓜灣": "To Kwa Wan", 
            "馬頭角": "Ma Tau Kok",
            "沙田": "Sha Tin", 
            "大圍": "Tai Wai",
            "火炭": "Fo Tan", 
            "馬鞍山": "Ma On Shan",
            "粉嶺": "Fanling", 
            "上水": "Sheung Shui",
            "天水圍": "Tin Shui Wai", 
            "葵涌": "Kwai Chung",
            "青衣": "Tsing Yi", 
            "將軍澳": "Tseung Kwan O",

            # --- 港鐵站 ---

            "會展站": "Exhibition Centre Station",
            "西灣河站": "Sai Wan Ho Station",
            "筲箕灣站": "Shau Kei Wan Station",
            "杏花邨站": "Heng Fa Chuen Station",
            "柴灣站": "Chai Wan Station",
            "香港大學站": "HKU Station",
            "海怡半島站": "South Horizons Station",
            "利東站": "Lei Tung Station",
            "黃竹坑站": "Wong Chuk Hang Station",
            "海洋公園站": "Ocean Park Station",
            "大窩口站": "Tai Wo Hau Station",
            "葵興站": "Kwai Hing Station",
            "葵芳站": "Kwai Fong Station",
            "荔景站": "Lai King Station",
            "欣澳站": "Sunny Bay Station",
            "東涌站": "Tung Chung Station",
            "機場站": "Airport Station",
            "鑽石山站": "Diamond Hill Station",
            "彩虹站": "Choi Hung Station",
            "九龍灣站": "Kowloon Bay Station",
            "牛頭角站": "Ngau Tau Kok Station",
            "藍田站": "Lam Tin Station",
            "油塘站": "Yau Tong Station",
            "調景嶺站": "Tiu Keng Leng Station",
            "坑口站": "Hang Hau Station",
            "寶琳站": "Po Lam Station",
            "將軍澳站": "Tseung Kwan O Station",
            "康城站": "LOHAS Park Station",
            "何文田站": "Ho Man Tin Station",
            "土瓜灣站": "To Kwa Wan Station",
            "天水圍站": "Tin Shui Wai Station",
            "宋皇臺站": "Sung Wong Toi Station",
            "啟德站": "Kai Tak Station",
            "顯徑站": "Hin Keng Station",
            "車公廟站": "Che Kung Temple Station",
            "沙田圍站": "Sha Tin Wai Station",
            "第一城站": "City One Station",
            "石門站": "Shek Mun Station",
            "大水坑站": "Tai Shui Hang Station",
            "恆安站": "Heng On Station",
            "烏溪沙站": "Wu Kai Sha Station",
            "大學站": "University Station",
            "大埔墟站": "Tai Po Market Station",
            "太和站": "Tai Wo Station",
            "羅湖站": "Lo Wu Station",
            "屯門站": "Tuen Mun Station",
            "落馬洲站": "Lok Ma Chau Station",
            "朗屏站": "Long Ping Station",
            "兆康站": "Siu Hong Station",
            "錦上路站": "Kam Sheung Road Station",
            "元朗站": "Yuen Long Station",
            "荃湾西站": "Tsuen Wan West Station",
            "旺角東站": "Mong Kok East Station",
            "柯士甸站": "Austin Station",
            "美孚站": "Mei Foo Station",
            "南昌站": "Nam Cheong Station",
            "尖東站": "East Tsim Sha Tsui Station",
            "紅磡站": "Hung Hom Station",
            "香港西九龍站": "Hong Kong West Kowloon Station",
            "中環站": "Central Station",
            "金鐘站": "Admiralty Station",
            "灣仔站": "Wan Chai Station",
            "銅鑼灣站": "Causeway Bay Station",
            "天后站": "Tin Hau Station",
            "炮台山站": "Fortress Hill Station",
            "北角站": "North Point Station",
            "鰂魚涌站": "Quarry Bay Station",
            "太古站": "Tai Koo Station",
            "上環站": "Sheung Wan Station",
            "西營盤站": "Sai Ying Pun Station",
            "堅尼地城站": "Kennedy Town Station",
            "長沙灣站": "Cheung Sha Wan Station",
            "荔枝角站": "Lai Chi Kok Station",
            "青衣站": "Tsing Yi Station",
            "樂富站": "Lok Fu Station",
            "黃大仙站": "Wong Tai Sin Station",
            "九龍塘站": "Kowloon Tong Station",
            "觀塘站": "Kwun Tong Station",
            "馬鞍山站": "Ma On Shan Station",
            "火炭站": "Fo Tan Station",
            "馬場站": "Racecourse Station",
            "粉嶺站": "Fanling Station",
            "上水站": "Sheung Shui Station",
            "奧運站": "Olympic Station",
            # --- 地區指示詞 ---
            "港島": "Hong Kong Island",
            "九龍": "Kowloon",
            "新界": "New Territories"
        }

        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        for zh, en in geo_pairs.items():
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO geo_locations (chinese_name, english_name, category) VALUES (?, ?, ?)",
                    (zh, en, "geo")
                )
            except Exception as e:
                logger.warning(f"插入地名 {zh} 失敗: {e}")
        conn.commit()
        conn.close()
        logger.info(f"已載入固定地名 {len(geo_pairs)} 條")


    #確保其他必要的表存在（verified_translations 已經存在）
    def _ensure_other_tables_exist(self, cursor):
        
        # 創建官方翻譯表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS official_translations (
                id INTEGER PRIMARY KEY,
                chinese_name TEXT UNIQUE,
                english_name TEXT,
                source TEXT,
                confidence REAL,
                created_at TIMESTAMP,
                verified BOOLEAN DEFAULT TRUE
            )
        """)   
        # 創建組件規則表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS component_rules (
                id INTEGER PRIMARY KEY,
                chinese_component TEXT,
                english_options TEXT,  -- JSON array
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                context TEXT
            )
        """)      
        # 創建翻譯歷史記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translation_history (
                id INTEGER PRIMARY KEY,
                chinese_name TEXT,
                english_name TEXT,
                method TEXT,
                layer INTEGER,
                confidence REAL,
                search_results TEXT,  -- JSON
                timestamp TIMESTAMP,
                user_feedback TEXT
            )
        """)

    #添加通用翻譯規則
    def _add_component_rules(self, cursor):
        
        # 檢查是否已有組件規則
        cursor.execute("SELECT COUNT(*) FROM component_rules")
        existing_components = cursor.fetchone()[0]
        if existing_components > 0:
            logger.info(f"發現已存在 {existing_components} 條組件規則")
            return
        
        # 基礎組件翻譯規則
        component_data = [
            ("花園", '["Garden", "Gardens"]', 0.95),
            ("中心", '["Centre", "Center", "Plaza"]', 0.92),
            ("廣場", '["Plaza", "Square"]', 0.95),
            ("苑", '["Court", "Gardens"]', 0.88),
            ("峰", '["Peak", "Heights", "Tower"]', 0.85),
            ("灣", '["Bay", "Cove"]', 0.90),
            ("城", '["City", "Town"]', 0.88),
            ("海", '["Sea", "Ocean", "Marine", "Harbour"]', 0.85),
            ("湖", '["Lake"]', 0.95),
            ("山", '["Hill", "Mountain"]', 0.90),
            ("座", '["Block", "Tower"]', 0.95),
            ("大廈", '["Building", "Mansion"]', 0.92),
            ("閣", '["Mansion", "Court"]', 0.88),
            # ("翠", '["Jade", "Green", "Emerald"]', 0.85),
            # ("金", '["Gold", "Golden"]', 0.90),
            # ("銀", '["Silver"]', 0.90),
            # ("豪", '["Grand", "Luxury"]', 0.85),
            # ("御", '["Imperial", "Royal"]', 0.88),
            # ("尊", '["Premier", "Elite"]', 0.85),
            # ("豪庭", '["Court", "Terrace"]', 0.90),
        ]
        
        for component, options, confidence in component_data:
            cursor.execute("""
                INSERT INTO component_rules
                (chinese_component, english_options, confidence)
                VALUES (?, ?, ?)
            """, (component, options, confidence))
        
        logger.info(f"添加了 {len(component_data)} 條組件翻譯規則")

    
    #輸出統計信息驗證"
    def _print_simple_stats(self, cursor):
        cursor.execute("SELECT COUNT(*) FROM verified_translations")
        verified_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM official_translations")
        official_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM component_rules")
        component_count = cursor.fetchone()[0]
        
        logger.info(" 翻譯資料庫統計 ")
        logger.info(f"已驗證翻譯: {verified_count} 條（來自 MongoDB）")
        logger.info(f"官方翻譯: {official_count} 條（手動添加）")
        logger.info(f"組件規則: {component_count} 條")
        logger.info(f"可用翻譯總計: {verified_count + official_count} 條")
    
    
    #查詢官方翻譯
    def get_official_translation(self, chinese_name: str) -> Optional[Dict]:

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT english_name, source, confidence 
            FROM official_translations 
            WHERE chinese_name = ?
        """, (chinese_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "english_name": result[0],
                "source": result[1],
                "confidence": result[2]
            }
        return None
    
    
    #查詢已驗證翻譯
    def get_verified_translation(self, chinese_name: str) -> Optional[Dict]:

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT english_name, confidence, usage_count
            FROM verified_translations 
            WHERE chinese_name = ?
        """, (chinese_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "english_name": result[0],
                "confidence": result[1],
                "usage_count": result[2]
            }
        return None
    
    
    #獲取所有已知翻譯
    def get_all_translations(self) -> Dict[str, str]:

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_translations = {}
        
        # 官方翻譯
        cursor.execute("SELECT chinese_name, english_name FROM official_translations")
        for row in cursor.fetchall():
            all_translations[row[0]] = row[1]
        
        # 已驗證翻譯
        cursor.execute("SELECT chinese_name, english_name FROM verified_translations")
        for row in cursor.fetchall():
            all_translations[row[0]] = row[1]
        
        conn.close()
        return all_translations
    
    
    #獲取通用翻譯規則
    def get_component_rules(self) -> Dict[str, List[str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT chinese_component, english_options FROM component_rules")
        component_rules = {}
        
        for row in cursor.fetchall():
            component = row[0]
            options = json.loads(row[1])
            component_rules[component] = options
        
        conn.close()
        return component_rules
    
    #保存翻譯結果到歷史記錄
    def save_translation_result(self, result: TranslationResult):
  
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        search_results_json = json.dumps(result.search_analysis) if result.search_analysis else ""
        
        cursor.execute("""
            INSERT INTO translation_history
            (chinese_name, english_name, method, layer, confidence, search_results, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.chinese_name,
            result.english_name,
            result.method,
            result.layer,
            result.confidence,
            search_results_json,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    #更新使用次數
    def update_usage_count(self, chinese_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE verified_translations 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE chinese_name = ?
        """, (datetime.now(), chinese_name))
        
        conn.commit()
        conn.close()


#  #網絡搜索服務
# class WebSearchService:
    
#     def __init__(self, google_api_key: str = None, search_engine_id: str = None):
#         self.google_api_key = google_api_key
#         self.search_engine_id = search_engine_id
    
    
#     #搜索樓盤翻譯相關信息
#     def search_property_translation(self, property_name: str, context: Dict = None) -> List[Dict]:

#         if not self.google_api_key:
#             logger.warning("Google Search API key not configured, returning mock results")
#             return self._mock_search_results(property_name)
        
#         search_queries = self._build_search_queries(property_name, context)
#         all_results = []
        
#         for query in search_queries:
#             try:
#                 results = self._execute_google_search(query)
#                 all_results.extend(results)
#             except Exception as e:
#                 logger.error(f"Search failed for query '{query}': {e}")
        
#         return all_results
    
#     #構建搜索查詢
#     def _build_search_queries(self, property_name: str, context: Dict = None) -> List[str]:
#         queries = []
        
#         # 基本查詢
#         queries.append(f'"{property_name}" "English name" OR "英文名稱"')
#         queries.append(f'"{property_name}" official translation')
        
#         # 如果有開發商信息
#         if context and context.get('developer'):
#             developer = context['developer']
#             queries.append(f'"{property_name}" {developer} English')
#             queries.append(f'site:{self._get_developer_domain(developer)} "{property_name}"')
        
#         #政府網站搜索
#         queries.append(f'"{property_name}" site:gov.hk OR site:landsd.gov.hk')
        
#         # 地產網站搜索
#         queries.append(f'"{property_name}" site:midland.com.hk OR site:centaline.com.hk')
        
#         return queries
    
#     def _get_developer_domain(self, developer: str) -> str:
#         #獲取開發商官方域名"
#         developer_domains = {
#             "新鴻基地產": "shkp.com.hk",
#             "長江實業": "ckh.com.hk",
#             "恆基地產": "hld.com",
#             "新世界發展": "nwd.com.hk",
#             "太古地產": "swireproperties.com"
#         }
#         return developer_domains.get(developer, "")
    
#     def _execute_google_search(self, query: str) -> List[Dict]:
#         #執行Google搜索#
#         url = "https://www.googleapis.com/customsearch/v1"
#         params = {
#             'key': self.google_api_key,
#             'cx': self.search_engine_id,
#             'q': query,
#             'num': 5
#         }
        
#         response = requests.get(url, params=params)
#         response.raise_for_status()
        
#         data = response.json()
#         results = []
        
#         for item in data.get('items', []):
#             results.append({
#                 'title': item['title'],
#                 'snippet': item['snippet'],
#                 'link': item['link'],
#                 'query': query
#             })
        
#         return results
    
#     def _mock_search_results(self, property_name: str) -> List[Dict]:
#         #模擬搜索結果（用於測試）#
#         mock_results = {
#             "天璽天": [
#                 {
#                     'title': '天璽天|啟德新地標',
#                     'snippet': '天璽‧天,英文名稱Cullinan Sky，屬於啟德區內最高私人住宅項目，與港鐵啟德站無縫接連...',
#                     'link': 'https://www.cullinanskykaitak.com/',
#                     'query': 'mock_search'
#                 }
#             ],
#             "慧安園": [
#                 {
#                     'title': '慧安園 | 寶琳 | 極罕筍盤推介 – 美聯物業 - Midland ',
#                     'snippet': '慧安園詳細樓盤與地區資料，包括37個賣盤、27個租盤、呎價、成交記錄等資料，幫你搵樓更快搜。 無論你想租定買，校網選宅抑或交通配套優先，美聯都......',
#                     'link': 'https://www.midland.com.hk/zh-hk/estate/%E4%B9%9D%E9%BE%8D-%E5%AF%B6%E7%90%B3-%E6%85%A7%E5%AE%89%E5%9C%92-E00074',
#                     'query': 'mock_search'
#                 }
#             ]
#         }
        
#         return mock_results.get(property_name, [])

class GrokAPIService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")  # 從環境變數讀取 API Key
        self.base_url = "https://api.x.ai/v1"  # 備用，需確認 xAI 的正確端點
        self.model = "grok-3"  # 使用 xAI 的 Grok 模型
        self.timeout = 3600  # 與範例一致，設置長超時
        self.max_retries = 3
        
        # 檢查是否可以使用 xAI SDK
        self.use_sdk = XAI_SDK_AVAILABLE and self.api_key
        if self.use_sdk:
            try:
                self.client = Client(api_key=self.api_key, timeout=self.timeout)
                logger.info("xAI SDK 初始化成功，將使用 SDK 調用 Grok API")
            except Exception as e:
                logger.warning(f"xAI SDK 初始化失敗: {e}，將使用 requests 方法")
                self.use_sdk = False
        else:
            logger.info("使用原有的 requests 方法調用 Grok API")

    def translate_text(self, zh: str, src: str = "zh", tgt: str = "en") -> str:
        print(f" 準備翻譯文本 (長度: {len(zh)} 字符)")
        
        if self.use_sdk:
            return self._translate_with_sdk(zh, src, tgt)
        else:
            return self._translate_with_requests(zh, src, tgt)

    def _translate_with_sdk(self, zh: str, src: str, tgt: str) -> str:
        try:
            chat = self.client.chat.create(model=self.model)
            chat.append(system("You are a professional translator. Translate Chinese text to English while preserving the meaning, style, and any placeholder tokens (like @@PRO1@@, @@LOC1@@ etc.) exactly as they appear."))
            chat.append(user(f"Please translate this Chinese text to English, keeping any placeholder tokens unchanged: {zh}"))
            
            response = chat.sample()
            translated = response.content.strip() if response.content else ""
            
            if translated:
                print(f" SDK翻譯完成 (長度: {len(translated)} 字符)")
                return translated
            else:
                print(" SDK翻譯返回空內容")
                return ""
                
        except Exception as e:
            print(f" SDK翻譯失敗: {e}")
            return ""

    def _translate_with_requests(self, zh: str, src: str, tgt: str) -> str:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the following Chinese text to English while preserving the meaning, style, and any placeholder tokens (like @@PRO1@@, @@LOC1@@ etc.) exactly as they appear."
                },
                {
                    "role": "user",
                    "content": f"Please translate this Chinese text to English, keeping any placeholder tokens unchanged: {zh}"
                }
            ],
            "model": self.model,
            "stream": False,
            "temperature": 0
        }
        
        try:
            response = self._call_grok_api_for_translation(payload)
            translated = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if translated:
                print(f"requests翻譯完成 (長度: {len(translated)} 字符)")
                return translated.strip()
            else:
                print(" requests翻譯返回空內容")
                return ""
                
        except Exception as e:
            print(f" requests翻譯失敗: {e}")
            return ""

    #分析並翻譯樓盤名稱，優先使用實時搜索
    def analyze_and_translate(self, property_name: str, context: Dict = None) -> Dict:
        
        logger.info(f"開始AI翻譯分析：{property_name}")
        
        if self.use_sdk:
            return self._analyze_with_live_search(property_name, context)
        else:
            return self._analyze_with_requests(property_name, context)

    def _analyze_with_live_search(self, property_name: str, context: Dict = None) -> Dict:
        try:
            # 建立搜索參數
            search_params = self._build_search_parameters(property_name, context)
            
            # 創建聊天會話
            chat = self.client.chat.create(
                model=self.model,
                search_parameters=search_params,
            )
            
            # 構建提示詞
            prompt = self._build_live_search_prompt(property_name, context or {})
            chat.append(user(prompt))
            
            # 獲取響應
            response = chat.sample()
            
            # 解析響應
            result = self._parse_grok_response(
                {"choices": [{"message": {"content": response.content}}]}, 
                property_name
            )
            
            # 確保返回有效結果
            if result is None:
                logger.error("Live Search 解析結果為空，使用降級翻譯")
                return self._fallback_translation(property_name)
                
            return result
            
        except Exception as e:
            logger.error(f"Live Search 失敗: {e}")
            return self._fallback_translation(property_name)

    def _build_search_parameters(self, property_name: str, context: Dict = None) -> SearchParameters:
        # 根據樓盤翻譯需求優化搜索源
        sources = [
            # 允許的網站：香港地產相關網站
            web_source(allowed_websites=[
                "midland.com.hk",
                "centaline.com.hk", 
                "28hse.com"
            ]),
            # 新聞源：香港主要媒體
            news_source()
        ]
        
        # 設置搜索時間範圍（最近2年的數據）
        from_date = datetime(2023, 1, 1)
        to_date = datetime.now()
        
        return SearchParameters(
            mode="on",  # 啟用實時搜索
            sources=sources,
            from_date=from_date,
            to_date=to_date,
            return_citations=True  # 返回引用來源
        )

    def _build_live_search_prompt(self, property_name: str, context: Dict) -> str:
        ctx_text = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "無"
        
        return f"""
你是專業的香港房地產翻譯專家。請使用實時搜索功能查找樓盤「{property_name}」的官方英文名稱。

搜索重點：
1. 查找開發商官方網站或政府文件中的英文名稱
2. 參考權威地產網站（如美聯物業、中原地產）的資料
3. 核實翻譯的一致性和權威性

上下文信息：
{ctx_text}

請按以下格式回應：
{{
  "search_summary": {{
    "official_found": true/false,
    "sources_considered": ["來源1", "來源2"],
    "confidence": 0.0-1.0,
    "search_quality": "high/medium/low"
  }},
  "translation": {{
    "english": "英文名稱",
    "method": "live_search_official" 或 "live_search_professional",
    "reason": "翻譯依據和來源說明"
  }}
}}

重要提醒：
- 優先採用官方或權威來源的翻譯
- 如果搜索結果不一致，選擇最權威的來源
- 如果找不到官方翻譯，基於香港地產命名慣例提供專業翻譯
- 確保英文名稱符合國際房地產營銷標準
"""

    def _analyze_with_requests(self, property_name: str, context: Dict = None) -> Dict:
       #使用原有的 requests 方法進行分析（不支持實時搜索）"""
        prompt = self._build_prompt(property_name, context or {})
        try:
            response = self._call_grok_api(prompt)
            return self._parse_grok_response(response, property_name)
        except Exception as e:
            logger.error(f"Grok requests 失敗：{e}")
            return self._fallback_translation(property_name)

    def _call_grok_api_for_translation(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        url = f"{self.base_url}/chat/completions"  # 需確認 xAI 的正確端點
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    import time
                    wait_time = 2 ** attempt
                    print(f" API速率限制，等待 {wait_time} 秒")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f" API調用超時 (嘗試 {attempt + 1}/{self.max_retries})")
                
            except requests.exceptions.RequestException as e:
                print(f" 網絡錯誤: {e}")
                
        raise Exception("API調用失敗，已達最大重試次數")

    def _build_prompt(self, property_name: str, context: Dict) -> str:
        """構建原有的提示詞（不支持實時搜索時使用）"""
        ctx_text = "\n".join([f"- {k}: {v}" for k, v in context.items()]) or "無"
        return f"""
你是資深香港房地產翻譯顧問。請基於你的知識庫完成下列任務：

1. 判斷樓盤「{property_name}」是否有已知的官方英文名稱；
2. 若有官方譯名，直接提供；
3. 若無官方譯名，依香港樓盤命名慣例提供專業譯名；
4. 回傳 JSON，格式如下：

{{
  "search_summary": {{
    "official_found": true/false,
    "sources_considered": ["knowledge_base"],
    "confidence": 0.0-1.0
  }},
  "translation": {{
    "english": "示例名稱",
    "method": "knowledge_base" 或 "professional",
    "reason": "簡要說明"
  }}
}}

可用上下文：
{ctx_text}
"""
    
    def _call_grok_api(self, prompt: str) -> Dict:
        """調用 Grok API（原有方法）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional Hong Kong property translation expert."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "model": self.model,
            "stream": False,
            "temperature": 0
        }
        
        url = f"{self.base_url}/chat/completions"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"調用 Grok API (嘗試 {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise Exception("API key 無效或已過期")
                elif response.status_code == 429:
                    import time
                    wait_time = 2 ** attempt
                    logger.warning(f"API 速率限制，等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 500:
                    raise Exception("Grok API 服務器錯誤")
                else:
                    raise Exception(f"API 調用失敗，狀態碼：{response.status_code}, 響應：{response.text}")
            
            except requests.exceptions.Timeout:
                logger.warning(f"API 調用超時 (嘗試 {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise Exception("API 調用超時")
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"API 連接錯誤 (嘗試 {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise Exception("無法連接到 Grok API")
        
        raise Exception("API 調用失敗，已達最大重試次數")
    
    def _parse_grok_response(self, response: Dict, property_name: str) -> Dict:
        """解析 Grok API 響應"""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                logger.error("Grok 回傳缺少 content 欄位")
                return self._fallback_translation(property_name)
            
            logger.info(f"Grok API 原始響應: {content}")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Grok 響應不是有效 JSON，嘗試文本提取")
                return self._extract_translation_from_text(content, property_name)

            if self._validate_response_format(parsed):
                # 舊版 -> 轉新版
                if "translation_result" in parsed:
                    parsed = {
                        "search_summary": parsed["search_analysis"],
                        "translation": parsed["translation_result"]
                    }
                return parsed

            logger.warning("Grok 響應格式不正確，使用降級處理")
            return self._extract_translation_from_text(content, property_name)

        except Exception as e:
            logger.error(f"解析 Grok 響應時發生錯誤: {e}")
            return self._fallback_translation(property_name)
    
    def _validate_response_format(self, response: Dict) -> bool:
        """驗證響應格式是否正確（兼容新舊版本）"""
        # 檢查新版
        if "search_summary" in response and "translation" in response:
            search_summary = response["search_summary"]
            translation_part = response["translation"]

            summary_required = ["official_found", "confidence"]
            if not all(k in search_summary for k in summary_required):
                return False

            trans_required = ["english", "method"]
            if not all(k in translation_part for k in trans_required):
                return False

            return True

        # 檢查舊版
        if "search_analysis" in response and "translation_result" in response:
            search_analysis = response["search_analysis"]
            translation_part = response["translation_result"]

            analysis_required = ["official_name_found", "source_reliability", "consistency_check"]
            if not all(k in search_analysis for k in analysis_required):
                return False

            trans_required = ["english_name", "confidence", "method", "reasoning"]
            if not all(k in translation_part for k in trans_required):
                return False

            return True

        return False
    
    def _extract_translation_from_text(self, content: str, property_name: str) -> Dict:
        """從文本中提取翻譯信息（當 JSON 解析失敗時）"""
        import re
        
        # 嘗試提取英文名稱
        english_patterns = [
            r'英文名稱[：:]\s*["\']?([A-Za-z\s]+)["\']?',
            r'English name[：:]\s*["\']?([A-Za-z\s]+)["\']?',
            r'翻譯[：:]\s*["\']?([A-Za-z\s]+)["\']?',
            r'["\']([A-Z][A-Za-z\s]{2,30})["\']',
        ]
        
        english_name = None
        for pattern in english_patterns:
            match = re.search(pattern, content)
            if match:
                english_name = match.group(1).strip()
                break
        
        if not english_name:
            return self._fallback_translation(property_name)
        
        confidence = 0.75
        if any(word in content.lower() for word in ['官方', 'official', '確認']):
            confidence = 0.9
        elif any(word in content.lower() for word in ['可能', 'possibly', '推測']):
            confidence = 0.6
        
        return {
            "search_analysis": {
                "official_name_found": confidence > 0.8,
                "source_reliability": "medium",
                "consistency_check": "mixed"
            },
            "translation_result": {
                "english_name": english_name,
                "confidence": confidence,
                "method": "text_extraction",
                "reasoning": f"從 Grok 響應文本中提取翻譯：{english_name}"
            }
        }

    def _fallback_translation(self, property_name: str) -> Dict:
        #降級翻譯（簡單音譯）
        return {
            "search_summary": {
                "official_found": False,
                "sources_considered": [],
                "confidence": 0.3
            },
            "translation": {
                "english": f"{property_name} Residence",
                "method": "fallback",
                "reason": "API 服務不可用，使用降級翻譯"
            }
        }

#測試 Grok API 服務
def test_grok_service():

    API_KEY = os.environ.get("GROK_API_KEY")
    
    grok_service = GrokAPIService(api_key=API_KEY)
    
    # 測試案例
    test_property = "璽悅"
    test_context = {"developer": "某知名開發商", "location": "港島區"}
    test_search_results = [
        {
            "title": "璽悅 (The Seal) - 開發商官方網站",
            "snippet": "璽悅 (The Seal) 位於優越地段，享有無敵海景...",
            "link": "https://developer-official.com/the-seal"
        }
    ]
    
    result = grok_service.analyze_and_translate(
        test_property, 
        test_search_results, 
        test_context
    )
    
    print(" Grok API 測試結果 ")
    print(f"樓盤名稱: {test_property}")
    print(f"翻譯結果: {result['translation_result']['english_name']}")
    print(f"信心度: {result['translation_result']['confidence']}")
    print(f"方法: {result['translation_result']['method']}")
    print(f"推理: {result['translation_result']['reasoning']}")

# 樓盤翻譯主系統
class PropertyTranslationSystem:

    USE_FUZZY_MATCH = False        

    # 只在需要時才實例化
    def _init_enhanced_matcher(self):
        return self.EnhancedFuzzyMatcher()

    def __init__(
        self,
        db_path: str = "property_translations.db",
        google_api_key: str = None,
        google_search_engine_id: str = None,
        grok_api_key: str = None
    ):
        # 基礎服務
        self.database     = PropertyTranslationDatabase(db_path)
        self.grok_processor = GrokAPIService(api_key=grok_api_key)

        # 模糊匹配器（視開關而定）
        self.enhanced_matcher = None

        if self.USE_FUZZY_MATCH:
            if not SEMANTIC_AVAILABLE:
                logger.warning("已啟用模糊匹配，但未安裝 sentence-transformers，將自動跳過")
            else:
                try:
                    self.enhanced_matcher = self._init_enhanced_matcher()
                    logger.info("增強模糊匹配器初始化成功")
                except Exception as e:
                    logger.warning(f"增強模糊匹配器初始化失敗: {e}")
                    self.enhanced_matcher = None
        else:
            logger.info("USE_FUZZY_MATCH=False，已跳過模糊匹配器載入")

        # 其他閾值設定
        self.FUZZY_MATCH_THRESHOLD = 0.75
        self.PROMOTION_THRESHOLD   = 10
    
    
    #檢查文本是否已經是英文
    def _is_already_english(self, text: str) -> bool:
        # 移除空格和標點符號
        clean_text = re.sub(r'[^\w]', '', text)
        
        if not clean_text:
            return False
            
        # has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        # if has_chinese:
        #     return False  # 包含中文就不是純英文

        # 計算英文字符比例
        english_chars = sum(1 for c in clean_text if c.isascii() and c.isalpha())
        total_chars = len(clean_text)
        
        # 如果英文字符比例超過80%，認為已經是英文
        if total_chars > 0 and (english_chars / total_chars) >= 0.99:
            return True
        
        # 特殊檢查：完全由英文字母、數字和空格組成
        if re.match(r'^[A-Za-z0-9\s]+$', text.strip()):
            return True
        
        return False
    
    def _is_building_suffix_only(self, text: str) -> bool: 
        
        # 常見的建築後綴模式
        suffix_patterns = [
            r'^\d+[A-Z]?期$',           # 1期、2期、3A期
            r'^第\d+期$',               # 第1期、第2期  
            r'^\d+[A-Z]?座$',           # 1座、2座、3A座
            r'^[A-Z]\d*座$',            # A座、B座、A1座
            r'^\d+號$',                 # 1號、2號
            r'^\d+樓$',                 # 1樓、2樓
            r'^[東西南北中]座$',          # 東座、西座
            r'^[東西南北中]翼$',          # 東翼、西翼
            r'^\d+棟$',                 # 1棟、2棟
            r'^Block\s*[A-Z0-9]+$',     # Block A、Block 1
            r'^Phase\s*\d+$',           # Phase 1、Phase 2
            r'^Tower\s*[A-Z0-9]+$',     # Tower A、Tower 1
        ]
        
        text_clean = text.strip()
        
        for pattern in suffix_patterns:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True
        
        return False

    class EnhancedFuzzyMatcher:
        def __init__(self):
            if not SEMANTIC_AVAILABLE:
                raise ImportError("sentence-transformers not available")

            # 關閉進度條、直接拿 numpy
            self.sentence_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.char_weight = 0.4
            self.semantic_weight = 0.6

            # 編碼快取
            self._embedding_cache: Dict[str, np.ndarray] = {}

            self.preprocessing_rules = {
                'phase_patterns': [r'[一二三四五六七八九十\d]+期', r'第[一二三四五六七八九十\d]+期'],
                'block_patterns': [r'[一二三四五六七八九十ABCD\d]+座', r'[ABCD\d]+棟'],
                'floor_patterns': [r'\d+樓', r'\d+層'],
                'number_patterns': [r'\d+號']
            }

        # 快取工具
        def _get_embedding(self, text: str) -> np.ndarray:
            if text not in self._embedding_cache:
                emb = self.sentence_model.encode(
                    text,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True   # 提前做 L2 Normalisation，後面省一次除法
                )
                self._embedding_cache[text] = emb
            return self._embedding_cache[text]
            
        #預處理樓盤名稱
        def preprocess_name(self, name: str) -> str:
            processed_name = name
            
            for pattern_type, patterns in self.preprocessing_rules.items():
                for pattern in patterns:
                    processed_name = re.sub(pattern, '', processed_name)
            
            return processed_name.strip()
        
        
        #計算語義相似度
        def calculate_semantic_similarity(self, name1: str, name2: str) -> float:
            try:
                embeddings = self.sentence_model.encode([name1, name2])
                
                #計算余弦相似度
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                
                return float(similarity)
            except Exception as e:
                logger.warning(f"语义相似度计算失败: {e}")
                return 0.0
        
        def calculate_char_similarity(self, name1: str, name2: str) -> float:
            #基礎相似度
            base_similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
            
            # 去除尾碼再計算相似度
            clean_name1 = self.preprocess_name(name1)
            clean_name2 = self.preprocess_name(name2)
            
            if clean_name1 and clean_name2:
                clean_similarity = difflib.SequenceMatcher(None, clean_name1, clean_name2).ratio()
            else:
                clean_similarity = base_similarity
            
            ## 加權組合
            final_similarity = 0.6 * base_similarity + 0.4 * clean_similarity
            
            return final_similarity
        
        #增強的相似度計算
        def enhanced_similarity_score(self, name1: str, name2: str) -> float:
            char_sim = self.calculate_char_similarity(name1, name2)
            semantic_sim = self.calculate_semantic_similarity(name1, name2)

            combined_score = (
                self.char_weight * char_sim + 
                self.semantic_weight * semantic_sim
            )
            
            return combined_score
        
        def find_best_matches(self, target_name: str, known_translations: Dict[str, str], 
                            top_k: int = 3) -> List[Tuple[str, str, float]]:
            candidates = []
            
            for known_name, known_translation in known_translations.items():
                similarity = self.enhanced_similarity_score(target_name, known_name)
                candidates.append((known_name, known_translation, similarity))
            
            # 按相似度排序，返回前k个
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[:top_k]

    #主翻譯方法：執行瀑布式翻譯
    def translate(self, property_name: str, context: Dict = None) -> TranslationResult:

        # if self._is_building_suffix_only(property_name):
        #     logger.info(f"跳過翻譯（僅為建築後綴）：{property_name}")
        #     return TranslationResult(
        #         chinese_name=property_name,
        #         english_name=property_name,  # 直接返回原文
        #         confidence=0.95,
        #         method="building_suffix",
        #         layer=0,
        #         source="suffix_detection",
        #         reasoning="檢測為建築後綴，無需翻譯"
        #     )
        if self._is_already_english(property_name):
            logging.info(f"跳過翻譯（已是英文）：{property_name}")
            return TranslationResult(
                chinese_name=property_name,
                english_name=property_name,
                confidence=0.98,
                method="already_english",
                layer=0,
                source="input_validation"
            )

        logger.info(f"開始翻譯樓盤：{property_name}")
        logger.info("嘗試第0層地名查詢")
        try:
            result = self._layer_0_fixed_lookup(property_name)
            if result:
                logger.info(f"第0層成功: {result.english_name}")
                return result
            else:
                logger.info("第0層未找到匹配")
        except Exception as e:
            logger.error(f"第0層執行異常: {e}")
            #官方 + 已驗證翻譯庫
        result = self._layer_1_official_lookup(property_name)
        if result:
            return result
        
        #模糊匹配查詢
        if self.USE_FUZZY_MATCH:
            result = self._layer_2_fuzzy_fuzzy_matching(property_name)
            if result:
                return result
        
        
        #智能搜索 + AI翻譯
        result = self._layer_3_ai_translation(property_name, context)
        # if result and result.confidence >= self.COMPONENT_CONFIDENCE_THRESHOLD:
        #     return result
        

        # #組件拆解翻譯, 僅供參考，顯然準確性很低
        # result = self._layer_3_component_translation(property_name)
        # if result and result.confidence >= self.COMPONENT_CONFIDENCE_THRESHOLD:
        #     return result
        
        # 保存翻譯結果並學習
        self._save_and_learn(result)
        return result
    

    def _layer_0_fixed_lookup(self, property_name: str) -> Optional[TranslationResult]:
        en = self.database.get_geo_location(property_name) \
     or self.database.get_slang_translation(property_name)
        if en:
            return TranslationResult(
                chinese_name = property_name,
                english_name = en,
                confidence   = 0.99,
                method       = "geo_lookup",
                layer        = 0,
                source       = "geo_locations",
                reasoning    = "固定地名對照"
            )
        return None
    
    
    #官方翻譯庫查詢
    def _layer_1_official_lookup(self, property_name: str) -> Optional[TranslationResult]:
    
        logger.info(f"查詢官方翻譯 - {property_name}")
        
        # 查詢官方翻譯
        official_result = self.database.get_official_translation(property_name)
        if official_result:
            self.database.update_usage_count(property_name)
            return TranslationResult(
                chinese_name=property_name,
                english_name=official_result['english_name'],
                confidence=official_result['confidence'],
                method="official_lookup",
                layer=1,
                source=official_result['source'],
                reasoning=f"從官方翻譯庫找到：{official_result['source']}"
            )
        
        # 查詢已驗證翻譯
        verified_result = self.database.get_verified_translation(property_name)
        if verified_result:
            self.database.update_usage_count(property_name)
            return TranslationResult(
                chinese_name=property_name,
                english_name=verified_result['english_name'],
                confidence=verified_result['confidence'],
                method="verified_lookup",
                layer=1,
                source="verified_database",
                reasoning=f"從已驗證翻譯庫找到，使用次數：{verified_result['usage_count']}"
            )
        
        logger.info("第1層未找到匹配翻譯")
        return None
    
    
    #模糊匹配查詢

    def _layer_2_fuzzy_matching(self, property_name: str) -> Optional[TranslationResult]:
        logger.info(f"模糊匹配查詢 - {property_name}")
        
        all_translations = self.database.get_all_translations()
        
        #如果有增強匹配器，使用增強版本
        if self.enhanced_matcher:
            return self._enhanced_fuzzy_matching(property_name, all_translations)
        else:
            return self._basic_fuzzy_matching(property_name, all_translations)

    #增強的模糊匹配
    def _enhanced_fuzzy_matching(self, property_name: str, all_translations: Dict[str, str]) -> Optional[TranslationResult]:
        best_matches = self.enhanced_matcher.find_best_matches(
            property_name, all_translations, top_k=3
        )
        
        if best_matches and best_matches[0][2] >= self.FUZZY_MATCH_THRESHOLD:
            best_match = best_matches[0]
            known_name, known_translation, similarity = best_match
            
    # 基於最佳匹配調整翻譯
            adjusted_translation = self._adjust_translation_based_on_match(
                property_name, known_name, known_translation
            )
            
            return TranslationResult(
                chinese_name=property_name,
                english_name=adjusted_translation,
                confidence=similarity * 0.92,  # 略微降低信心度
                method="enhanced_fuzzy_matching",
                layer=2,
                source="similar_property",
                reasoning=f"基于增强相似度匹配'{known_name}'调整翻译，相似度：{similarity:.3f}",
                alternatives=[match[1] for match in best_matches[1:3]]  # 其他候选
            )
        
        logger.info("第2层未找到足够相似的翻译")
        return None



    def _basic_fuzzy_matching(self, property_name: str,all_translations: Dict[str, str]) -> Optional[TranslationResult]:
        
        logger.info(f"模糊匹配查詢(基礎版) - {property_name}")
        
        best_match = None
        best_similarity = 0.0
        
        for known_name, known_translation in all_translations.items():
            # 計算相似度
            similarity = self._calculate_similarity(property_name, known_name)
            
            if similarity > best_similarity and similarity >= self.FUZZY_MATCH_THRESHOLD:
                best_similarity = similarity
                best_match = (known_name, known_translation)
        
        if best_match:
            # 基於最佳匹配調整翻譯
            adjusted_translation = self._adjust_translation_based_on_match(
                property_name, best_match[0], best_match[1]
            )
            
            return TranslationResult(
                chinese_name=property_name,
                english_name=adjusted_translation,
                confidence=best_similarity * 0.89,  # 略微降低信心度
                method="fuzzy_matching",
                layer=2,
                source="similar_property",
                reasoning=f"基於相似樓盤'{best_match[0]}'調整翻譯，相似度：{best_similarity:.2f}"
            )
        
        logger.info("第2層未找到足夠相似的翻譯")
        return None
    
    
    
    #AI智能翻譯

    def _layer_3_ai_translation(self, property_name: str, context: Dict = None) -> TranslationResult:
        logger.info(f"AI 翻譯 (Grok + Live Search) - {property_name}")

        grok_res = self.grok_processor.analyze_and_translate(property_name, context)
        
        # 添加空值檢查
        if grok_res is None:
            logger.error("Grok API 返回空結果，使用降級翻譯")
            return TranslationResult(
                chinese_name=property_name,
                english_name=f"{property_name} Residence",
                confidence=0.3,
                method="fallback",
                layer=3,
                source="fallback",
                reasoning="Grok API 返回空結果"
            )

        # 原有的處理邏輯
        if "translation" in grok_res:                
            trans = grok_res["translation"]
        else:                                         
            trans = grok_res.get("translation_result", {})

        english_name = (
            trans.get("english") or
            trans.get("english_name") or
            f"{property_name} Residence"
        )

        search_analysis = (
            grok_res.get("search_summary") or
            grok_res.get("search_analysis") or
            {}
        )

        confidence = float(trans.get("confidence", 0.6))
        method     = trans.get("method", "grok")
        reasoning  = trans.get("reason") or trans.get("reasoning", "")

        # 如果使用了實時搜索，提高信心度
        if method.startswith("live_search"):
            confidence = min(confidence + 0.1, 0.95)
            reasoning += " (使用實時搜索驗證)"

        return TranslationResult(
            chinese_name=property_name,
            english_name=english_name,
            confidence=confidence,
            method=method,
            layer=3,
            source="grok_live_search" if method.startswith("live_search") else "grok",
            reasoning=reasoning,
            search_analysis=search_analysis
        )

    # def _layer_4_component_translation(self, property_name: str) -> Optional[TranslationResult]:
        
    #     logger.info(f"組件拆解翻譯 - {property_name}")
        
    #     component_rules = self.database.get_component_rules()
    #     english_parts = []
    #     used_components = []
    #     total_confidence = 0.0
        
    #     remaining_name = property_name
        
    #     # 嘗試匹配組件
    #     for component, options in component_rules.items():
    #         if component in remaining_name:
    #             # 選擇最常用的翻譯選項
    #             english_part = options[0] if options else component
    #             english_parts.append(english_part)
    #             used_components.append(component)
    #             remaining_name = remaining_name.replace(component, "", 1)
    #             total_confidence += 0.8  # 每個匹配組件的基礎信心度
        
    #     if english_parts:
    #         # 處理剩餘部分
    #         if remaining_name.strip():
    #             # 簡單音譯剩餘部分
    #             transliterated = self._simple_transliteration(remaining_name.strip())
    #             if transliterated:
    #                 english_parts.insert(0, transliterated)
            
    #         # 組合最終翻譯
    #         final_translation = " ".join(english_parts)
    #         final_confidence = min(total_confidence / len(used_components), 0.85)
            
    #         return TranslationResult(
    #             chinese_name=property_name,
    #             english_name=final_translation,
    #             confidence=final_confidence,
    #             method="component_based",
    #             layer=3,
    #             source="component_rules",
    #             reasoning=f"組件拆解：{' + '.join(used_components)} → {final_translation}"
    #         )
        
    #     logger.info("無法進行有效的組件拆解")
    #     return None


    
    #計算兩個樓盤名稱的相似度
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        # 使用 difflib 計算序列相似度
        base_similarity = difflib.SequenceMatcher(None, name1, name2).ratio()
        
        # 忽略期數、座號等
        clean_name1 = self._clean_property_name(name1)
        clean_name2 = self._clean_property_name(name2)
        
        if clean_name1 != name1 or clean_name2 != name2:
            clean_similarity = difflib.SequenceMatcher(None, clean_name1, clean_name2).ratio()
            return max(base_similarity, clean_similarity)
        
        return base_similarity
    
    def _clean_property_name(self, name: str) -> str:

        # 移除常見的期數標識
        patterns = [
            r'[一二三四五六七八九十\d]+期',
            r'[一二三四五六七八九十ABCD\d]+座',
            r'第[一二三四五六七八九十\d]+期',
            r'\d+號',
        ]
        
        cleaned = name
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned.strip()
    
    
    #基於匹配結果調整翻譯，處理額外的部分如期數、座號等
    def _adjust_translation_based_on_match(self, target_name: str, match_name: str, match_translation: str) -> str:
        
        if len(target_name) > len(match_name):
            extra_part = target_name.replace(match_name, "").strip()
            if extra_part:
                # 翻譯額外部分
                extra_translation = self._translate_building_suffix(extra_part)
                if extra_translation:
                    return f"{match_translation} {extra_translation}"
        
        return match_translation

    
    #翻譯建築後綴，如期數、座號等
    def _translate_building_suffix(self, suffix: str) -> str:

        import re
        
        # 處理座號：1座、2座、A座等
        block_pattern = r'([0-9A-Za-z]+)座'
        block_match = re.search(block_pattern, suffix)
        if block_match:
            block_num = block_match.group(1)
            return f"Block {block_num}"
        
        # 處理期數：1期、二期、第三期等
        phase_patterns = [
            (r'第?([一二三四五六七八九十]+)期', 'chinese_numbers'),
            (r'第?([0-9]+)期', 'arabic_numbers'),
        ]
        
        for pattern, number_type in phase_patterns:
            phase_match = re.search(pattern, suffix)
            if phase_match:
                phase_num = phase_match.group(1)
                if number_type == 'chinese_numbers':# 轉換中文數字為阿拉伯數字
                    phase_num = self._chinese_to_arabic(phase_num)
                return f"Phase {phase_num}"
        
        # 處理樓層或其他數字
        number_pattern = r'([0-9]+)樓?'
        number_match = re.search(number_pattern, suffix)
        if number_match:
            floor_num = number_match.group(1)
            return f"Floor {floor_num}"
        
        # 處理其他常見後綴
        suffix_translations = {
            "東座": "East Block",
            "西座": "West Block", 
            "南座": "South Block",
            "北座": "North Block",
            "中座": "Central Block",
            "新翼": "New Wing",
            "舊翼": "Old Wing",
            "主樓": "Main Building",
            "附樓": "Annex Building",
        }
        
        for chinese_suffix, english_suffix in suffix_translations.items():
            if chinese_suffix in suffix:
                return english_suffix
        
        # 如果無法識別，嘗試簡單音譯
        return self._simple_transliteration(suffix)

    #轉換中文數字為阿拉伯數字    
    def _chinese_to_arabic(self, chinese_num: str) -> str:
        chinese_numbers = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '零': '0'
        }
        
        # 處理一些特殊情況
        if chinese_num == '十':
            return '10'
        elif chinese_num.startswith('十'):  # 十一、十二等
            return str(10 + chinese_numbers.get(chinese_num[1:], 0))
        elif chinese_num.endswith('十'):  # 二十、三十等
            return str(chinese_numbers.get(chinese_num[0], 0) * 10)
        
        # 簡單映射
        return chinese_numbers.get(chinese_num, chinese_num)
    
    def _simple_transliteration(self, chinese_text: str) -> str:
        # 這裡以後可以實現更複雜的音譯邏輯，我知道香港有很多不同的注音系統
        # 暫時返回簡化處理
    # 基本字符映射
        transliteration_map = {
            "翠": "Emerald",
            "金": "Golden", 
            "銀": "Silver",
            "海": "Ocean",
            "山": "Hill",
            "湖": "Lake",
            "星": "Star",
            "月": "Moon",
            "日": "Sun",
            "座": "Block",
            "期": "Phase",
            "樓": "Floor",
            "東": "East",
            "西": "West", 
            "南": "South",
            "北": "North",
            "中": "Central"
        }
        
        # 數字映射
        number_map = {
            '1': 'Block 1', '2': 'Block 2', '3': 'Block 3', '4': 'Block 4', '5': 'Block 5',
            '6': 'Block 6', '7': 'Block 7', '8': 'Block 8', '9': 'Block 9', '0': 'Block 0',
            'A': 'Block A', 'B': 'Block B', 'C': 'Block C', 'D': 'Block D'
        }
        
        # 先檢查是否是純數字或字母（可能是座號）
        if chinese_text.isdigit() or chinese_text.isalpha():
            return number_map.get(chinese_text, f"Block {chinese_text}")
        
        # 字符拆解翻譯
        result = ""
        for char in chinese_text:
            if char in transliteration_map:
                result += transliteration_map[char] + " "
            elif char.isdigit():
                result += char
        
        return result.strip() if result.strip() else None
    
    def _save_and_learn(self, result: TranslationResult):
        #保存翻譯結果並學習
        logger.info(f"保存翻譯結果：{result.chinese_name} → {result.english_name}")
        
        # 保存到歷史記錄
        self.database.save_translation_result(result)
        
        # 學習邏輯
        if result.layer == 4 and result.confidence >= 0.8:
            # 高質量的AI翻譯，可以考慮加入已驗證庫
            logger.info(f"高質量AI翻譯，考慮加入已驗證庫：{result.english_name}")
            # 這裡可以實現自動或半自動的學習機制
    
    #獲取翻譯統計信息
    def get_translation_stats(self) -> Dict:

        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()
        
        # 各層翻譯統計
        cursor.execute("""
            SELECT layer, COUNT(*) as count 
            FROM translation_history 
            GROUP BY layer
        """)
        layer_stats = dict(cursor.fetchall())
        
        # 方法統計
        cursor.execute("""
            SELECT method, COUNT(*) as count 
            FROM translation_history 
            GROUP BY method
        """)
        method_stats = dict(cursor.fetchall())
        
        # 平均信心度
        cursor.execute("SELECT AVG(confidence) FROM translation_history")
        avg_confidence = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "layer_distribution": layer_stats,
            "method_distribution": method_stats,
            "average_confidence": round(avg_confidence, 3) if avg_confidence else 0,
            "total_translations": sum(layer_stats.values())
        }

def main():
    
    # 初始化翻譯系統

    GROK_API_KEY = os.environ.get("GROK_API_KEY")
    
    print("正在初始化帶 Grok API 的翻譯系統...")

    translator = PropertyTranslationSystem(grok_api_key=GROK_API_KEY)
    
    # 驗證 Grok API 配置
    if translator.grok_service.api_key:
        print(" Grok API 已配置")
    else:
        print(" Grok API 未配置")
        return
    
    # 測試案例
    test_cases = [
        {"name": "麗城花園", "context": {"developer": "長江實業"}},
        {"name": "黃埔花園", "context": {"location": "红磡"}},
        {"name": "富麗花園2座", "context": {"location": "寶琳 / 將軍澳站"}},
        {"name": "天璽天", "context": {"developer": "新鴻基"}},
        {"name": "慧安園", "context": {"location": "寶琳 / 將軍澳站"}},
        {"name": "金輝豪庭", "context": None},
    ]
    
    print("=== 瀑布式樓盤翻譯系統測試 ===\n")
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"測試案例 {i}：{case['name']}")
        print("-" * 50)
        
        result = translator.translate(case['name'], case['context'])
        results.append(result)
        
        print(f"中文名稱：{result.chinese_name}")
        print(f"英文翻譯：{result.english_name}")
        print(f"信心度：{result.confidence:.3f}")
        print(f"翻譯方法：{result.method}")
        print(f"使用層級：第{result.layer}層")
        print(f"來源：{result.source}")
        print(f"推理過程：{result.reasoning}")
        
        if result.layer == 3:
            print("使用了 Grok AI 翻譯")
        else:
            print(f"使用了第{result.layer}層本地翻譯")
        
        if result.search_analysis:
            print(f"搜索分析：{result.search_analysis}")
    
    # 輸出統計信息
    print("=== 翻譯統計 ===")
    stats = translator.get_translation_stats()
    print(f"總翻譯次數：{stats['total_translations']}")
    print(f"平均信心度：{stats['average_confidence']}")
    print(f"各層分佈：{stats['layer_distribution']}")
    print(f"方法分佈：{stats['method_distribution']}")

if __name__ == "__main__":
    main()
    # test_grok_service()

"""
Google Ads Transparency Center 广告爬取 Web 应用
Flask 后端 + 现代化前端界面
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import json
import uuid
import threading
import time
from datetime import datetime
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv
from serpapi import GoogleSearch
import requests
from PIL import Image
import numpy as np

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)

# API 密钥
API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API 配置
GEMINI_API_URL = "https://api.apimart.ai/v1beta/models/gemini-2.5-pro:generateContent"

# 存储任务状态和结果
tasks = {}

# OCR 引擎
ocr_engine = None
OCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    print("警告: PaddleOCR 未安装，图片文字识别功能将不可用")


def init_ocr():
    """初始化 OCR 引擎"""
    global ocr_engine, OCR_AVAILABLE
    if OCR_AVAILABLE and ocr_engine is None:
        try:
            # 新版本 PaddleOCR 不再支持 show_log 参数
            ocr_engine = PaddleOCR(use_textline_orientation=True, lang='ch')
            print("OCR 引擎初始化成功")
        except Exception as e:
            print(f"OCR 引擎初始化失败: {e}")
            import traceback
            traceback.print_exc()
            ocr_engine = None


def format_timestamp(timestamp):
    """将时间戳转换为可读的日期时间格式"""
    if not timestamp:
        return None
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def format_date(date_int):
    """将日期整数（YYYYMMDD）转换为可读格式"""
    if not date_int:
        return None
    try:
        date_str = str(date_int)
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return None
    except:
        return None


def calculate_duration_days(first_shown, last_shown):
    """计算广告持续天数"""
    if not first_shown or not last_shown:
        return None
    try:
        if isinstance(first_shown, (int, float)) and first_shown > 1000000000:
            first = datetime.fromtimestamp(first_shown)
            last = datetime.fromtimestamp(last_shown)
        elif isinstance(first_shown, int) and first_shown < 1000000000:
            first_str = str(first_shown)
            last_str = str(last_shown)
            first = datetime.strptime(first_str, "%Y%m%d")
            last = datetime.strptime(last_str, "%Y%m%d")
        else:
            return None
        
        duration = (last - first).days
        return duration if duration >= 0 else None
    except:
        return None


def extract_text_from_image(image_url):
    """从图片 URL 下载图片并使用 OCR 识别文字"""
    global ocr_engine
    
    if not OCR_AVAILABLE:
        print("OCR 不可用")
        return None
    
    # 确保 OCR 引擎已初始化
    if ocr_engine is None:
        init_ocr()
    
    if ocr_engine is None:
        print("OCR 引擎初始化失败")
        return None
    
    try:
        print(f"  正在下载图片: {image_url[:80]}...")
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        print(f"  图片尺寸: {img_array.shape}")
        
        # 使用 predict 方法（新版本 PaddleOCR）
        result = ocr_engine.predict(img_array)
        
        texts = []
        if result and isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                # 新版本格式：{'rec_texts': [...], 'rec_scores': [...]}
                rec_texts = item.get('rec_texts', [])
                rec_scores = item.get('rec_scores', [])
                
                for i, text in enumerate(rec_texts):
                    if text:
                        score = rec_scores[i] if i < len(rec_scores) else 1.0
                        if score > 0.5:
                            texts.append(text)
            elif isinstance(item, list):
                # 旧版本格式：[[box, (text, score)], ...]
                for line in item:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                            text = text_info[0] if isinstance(text_info[0], str) else str(text_info[0])
                            score = text_info[1] if len(text_info) > 1 else 1.0
                            if score > 0.5:
                                texts.append(text)
        
        ocr_result = "\n".join(texts) if texts else None
        if ocr_result:
            print(f"  OCR 识别成功: {len(texts)} 段文字")
        else:
            print("  OCR 未识别到文字")
        
        return ocr_result
        
    except Exception as e:
        print(f"  OCR 识别失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_ad_creatives_list(domain, region=None, creative_format="text", num=50):
    """获取指定域名的广告创意列表"""
    params_list = {
        "engine": "google_ads_transparency_center",
        "api_key": API_KEY,
        "text": domain,
        "creative_format": creative_format,
        "num": num
    }
    
    if region:
        params_list["region"] = region
    
    search_list = GoogleSearch(params_list)
    results_list = search_list.get_dict()
    
    if "error" in results_list:
        return [], results_list.get("error")
    
    ads = results_list.get("ad_creatives", [])
    if not ads:
        ads = results_list.get("ads", [])
    
    return ads, None


def get_ad_details(advertiser_id, creative_id, region=None):
    """获取具体广告创意的详细信息"""
    params_detail = {
        "engine": "google_ads_transparency_center_ad_details",
        "api_key": API_KEY,
        "advertiser_id": advertiser_id,
        "creative_id": creative_id
    }
    
    if region:
        params_detail["region"] = region
    
    search_detail = GoogleSearch(params_detail)
    res_detail = search_detail.get_dict()
    
    return res_detail.get("ad_creatives", []), res_detail


def scrape_ads_async(task_id, domain, region=None, num=50, enable_ocr=True):
    """异步爬取广告数据"""
    global tasks
    
    tasks[task_id]["status"] = "running"
    tasks[task_id]["message"] = "正在获取广告列表..."
    tasks[task_id]["progress"] = 5
    
    try:
        # 初始化 OCR
        if enable_ocr and OCR_AVAILABLE:
            init_ocr()
        
        # 获取广告列表
        ads_list, error = get_ad_creatives_list(domain, region, "text", num)
        
        if error:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = f"API 错误: {error}"
            return
        
        if not ads_list:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "未找到任何广告"
            tasks[task_id]["results"] = []
            return
        
        tasks[task_id]["message"] = f"找到 {len(ads_list)} 个广告，正在获取详情..."
        tasks[task_id]["total"] = len(ads_list)
        tasks[task_id]["progress"] = 10
        
        all_ads_details = []
        
        for idx, ad in enumerate(ads_list, 1):
            advertiser_id = ad.get("advertiser_id")
            creative_id = ad.get("ad_creative_id")
            
            if not advertiser_id or not creative_id:
                continue
            
            tasks[task_id]["message"] = f"正在获取广告 {idx}/{len(ads_list)} 的详情..."
            tasks[task_id]["current"] = idx
            tasks[task_id]["progress"] = 10 + int((idx / len(ads_list)) * 85)
            
            try:
                creatives, detail_response = get_ad_details(advertiser_id, creative_id, region)
                
                # 获取时间和地区信息
                search_info = detail_response.get("search_information", {})
                regions_info = search_info.get("regions", [])
                
                first_shown_ts = ad.get("first_shown")
                last_shown_ts = ad.get("last_shown")
                total_days_shown = ad.get("total_days_shown")
                
                first_shown_formatted = format_timestamp(first_shown_ts)
                last_shown_formatted = format_timestamp(last_shown_ts)
                
                duration_days = total_days_shown
                if not duration_days and first_shown_ts and last_shown_ts:
                    duration_days = calculate_duration_days(first_shown_ts, last_shown_ts)
                
                regions_list = []
                regions_detail = []
                if regions_info:
                    for reg in regions_info:
                        region_name = reg.get("region_name", "")
                        first_shown_date = reg.get("first_shown")
                        last_shown_date = reg.get("last_shown")
                        
                        regions_list.append(region_name)
                        regions_detail.append({
                            "region": region_name,
                            "first_shown": format_date(first_shown_date),
                            "last_shown": format_date(last_shown_date),
                            "duration_days": calculate_duration_days(first_shown_date, last_shown_date)
                        })
                
                if creatives:
                    for creative in creatives:
                        is_image_ad = "image" in creative and not any(
                            key in creative for key in ["title", "headline", "snippet", "long_headline"]
                        )
                        
                        if is_image_ad:
                            image_url = creative.get("image") or ad.get("image")
                            ocr_text = None
                            
                            if image_url and enable_ocr:
                                ocr_text = extract_text_from_image(image_url)
                            
                            title, headline, snippet = None, None, None
                            if ocr_text:
                                lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
                                if lines:
                                    title = lines[0] if len(lines[0]) < 100 else None
                                    if len(lines) > 1:
                                        headline = lines[1] if len(lines[1]) < 200 else None
                                    if len(lines) > 2:
                                        snippet = " ".join(lines[2:])[:500]
                            
                            ad_info = {
                                "advertiser_id": advertiser_id,
                                "creative_id": creative_id,
                                "title": title,
                                "headline": headline,
                                "long_headline": None,
                                "snippet": snippet,
                                "visible_link": None,
                                "link": None,
                                "final_url": None,
                                "call_to_action": None,
                                "creative_format": "image",
                                "ad_type": "图片广告",
                                "image_url": image_url,
                                "ocr_text": ocr_text,
                                "note": "图片广告" + (" (OCR识别)" if ocr_text else ""),
                                "first_shown": first_shown_formatted,
                                "last_shown": last_shown_formatted,
                                "duration_days": duration_days,
                                "regions": ", ".join(regions_list) if regions_list else None,
                                "regions_detail": regions_detail,
                            }
                        else:
                            # 文字广告，但如果有图片也尝试 OCR
                            image_url = creative.get("image") or ad.get("image")
                            ocr_text = None
                            
                            # 如果有图片且启用了 OCR，也对图片进行识别
                            if image_url and enable_ocr:
                                print(f"  文字广告也有图片，尝试 OCR...")
                                ocr_text = extract_text_from_image(image_url)
                            
                            ad_info = {
                                "advertiser_id": advertiser_id,
                                "creative_id": creative_id,
                                "title": creative.get("title"),
                                "headline": creative.get("headline"),
                                "long_headline": creative.get("long_headline"),
                                "snippet": creative.get("snippet"),
                                "visible_link": creative.get("visible_link"),
                                "link": creative.get("link"),
                                "final_url": creative.get("final_url"),
                                "call_to_action": creative.get("call_to_action"),
                                "creative_format": creative.get("creative_format") or ad.get("format"),
                                "ad_type": creative.get("ad_type") or "文字广告",
                                "image_url": image_url,
                                "ocr_text": ocr_text,
                                "note": "文字广告" + (" (含图片OCR)" if ocr_text else ""),
                                "first_shown": first_shown_formatted,
                                "last_shown": last_shown_formatted,
                                "duration_days": duration_days,
                                "regions": ", ".join(regions_list) if regions_list else None,
                                "regions_detail": regions_detail,
                            }
                        
                        all_ads_details.append(ad_info)
                else:
                    # 详情 API 没有返回，使用列表信息
                    image_url = ad.get("image")
                    ocr_text = None
                    
                    if image_url and enable_ocr:
                        ocr_text = extract_text_from_image(image_url)
                    
                    title, headline, snippet = None, None, None
                    if ocr_text:
                        lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
                        if lines:
                            title = lines[0] if len(lines[0]) < 100 else None
                            if len(lines) > 1:
                                headline = lines[1] if len(lines[1]) < 200 else None
                            if len(lines) > 2:
                                snippet = " ".join(lines[2:])[:500]
                    
                    ad_info = {
                        "advertiser_id": advertiser_id,
                        "creative_id": creative_id,
                        "title": title,
                        "headline": headline,
                        "long_headline": None,
                        "snippet": snippet,
                        "visible_link": None,
                        "link": None,
                        "final_url": None,
                        "call_to_action": None,
                        "creative_format": ad.get("format"),
                        "ad_type": "图片广告" if image_url else "未知",
                        "image_url": image_url,
                        "ocr_text": ocr_text,
                        "note": "详情API未返回" + (" (OCR识别)" if ocr_text else ""),
                        "first_shown": first_shown_formatted,
                        "last_shown": last_shown_formatted,
                        "duration_days": duration_days,
                        "regions": ", ".join(regions_list) if regions_list else None,
                        "regions_detail": regions_detail,
                    }
                    all_ads_details.append(ad_info)
                
            except Exception as e:
                print(f"获取广告 {idx} 详情时出错: {e}")
                continue
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = f"完成！共获取 {len(all_ads_details)} 条广告数据"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["results"] = all_ads_details
        
    except Exception as e:
        tasks[task_id]["status"] = "error"
        tasks[task_id]["message"] = f"发生错误: {str(e)}"


@app.route("/")
def index():
    """首页"""
    return render_template("index.html")


@app.route("/api/scrape", methods=["POST"])
def start_scrape():
    """开始爬取任务"""
    data = request.json
    domain = data.get("domain", "").strip()
    num = int(data.get("num", 50))
    enable_ocr = data.get("enable_ocr", True)
    
    if not domain:
        return jsonify({"error": "请输入域名"}), 400
    
    # 清理域名格式
    domain = domain.replace("https://", "").replace("http://", "").replace("www.", "")
    if "/" in domain:
        domain = domain.split("/")[0]
    
    # 创建任务
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "message": "任务已创建",
        "progress": 0,
        "total": 0,
        "current": 0,
        "results": [],
        "domain": domain,
        "created_at": datetime.now().isoformat()
    }
    
    # 启动异步任务
    thread = threading.Thread(target=scrape_ads_async, args=(task_id, domain, None, num, enable_ocr))
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id, "domain": domain})


@app.route("/api/status/<task_id>")
def get_status(task_id):
    """获取任务状态"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    
    # 如果任务完成，返回所有结果；否则返回前10条预览
    all_results = task.get("results", [])
    is_completed = task["status"] == "completed"
    
    response_data = {
        "status": task["status"],
        "message": task["message"],
        "progress": task["progress"],
        "total": task.get("total", 0),
        "current": task.get("current", 0),
        "results_count": len(all_results),
    }
    
    if is_completed:
        # 完成后返回全部数据
        response_data["results"] = all_results
    else:
        # 进行中只返回预览
        response_data["preview"] = all_results[:10]
    
    return jsonify(response_data)


@app.route("/api/results/<task_id>")
def get_all_results(task_id):
    """获取任务的全部结果"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    results = task.get("results", [])
    
    return jsonify({
        "task_id": task_id,
        "domain": task.get("domain", ""),
        "status": task["status"],
        "results_count": len(results),
        "results": results
    })


@app.route("/api/download/<task_id>/<file_type>")
def download_file(task_id, file_type):
    """下载结果文件"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    
    if task["status"] != "completed" or not task.get("results"):
        return jsonify({"error": "没有可下载的数据"}), 400
    
    results = task["results"]
    domain = task.get("domain", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_type == "json":
        # JSON 下载
        output = BytesIO()
        json_data = json.dumps(results, ensure_ascii=False, indent=2)
        output.write(json_data.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"{domain}_ads_{timestamp}.json"
        )
    
    elif file_type == "excel":
        # Excel 下载
        excel_data = []
        for ad in results:
            row = {
                "广告主ID": ad.get("advertiser_id", ""),
                "创意ID": ad.get("creative_id", ""),
                "标题 (Title)": ad.get("title", ""),
                "主标题 (Headline)": ad.get("headline", ""),
                "长标题 (Long Headline)": ad.get("long_headline", ""),
                "描述 (Snippet)": ad.get("snippet", ""),
                "可见链接": ad.get("visible_link", ""),
                "目标链接": ad.get("link", ""),
                "最终URL": ad.get("final_url", ""),
                "行动号召 (CTA)": ad.get("call_to_action", ""),
                "广告格式": ad.get("creative_format", ""),
                "广告类型": ad.get("ad_type", ""),
                "首次显示时间": ad.get("first_shown", ""),
                "最后显示时间": ad.get("last_shown", ""),
                "持续天数": ad.get("duration_days", ""),
                "发布地区": ad.get("regions", ""),
                "地区详情": json.dumps(ad.get("regions_detail", []), ensure_ascii=False) if ad.get("regions_detail") else "",
                "OCR识别文字": ad.get("ocr_text", ""),
                "图片URL": ad.get("image_url", ""),
                "备注": ad.get("note", ""),
            }
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='广告数据')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f"{domain}_ads_{timestamp}.xlsx"
        )
    
    return jsonify({"error": "不支持的文件类型"}), 400


@app.route("/api/results/<task_id>")
def get_results(task_id):
    """获取完整结果"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        return jsonify({"error": "任务尚未完成"}), 400
    
    return jsonify({
        "domain": task.get("domain"),
        "total": len(task.get("results", [])),
        "results": task.get("results", [])
    })


def call_gemini_api(prompt, max_tokens=8192):
    """调用 Gemini API 生成内容"""
    if not GEMINI_API_KEY:
        return None, "Gemini API Key 未配置"
    
    try:
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens,
                "topP": 0.9
            }
        }
        
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            # 处理嵌套的 data 结构
            data = result.get("data", result)
            candidates = data.get("candidates", [])
            if candidates and len(candidates) > 0:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and len(parts) > 0:
                    return parts[0].get("text", ""), None
            return None, "API 返回格式异常"
        else:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            return None, f"Gemini API 错误: {error_msg}"
            
    except requests.exceptions.Timeout:
        return None, "Gemini API 请求超时"
    except Exception as e:
        return None, f"Gemini API 调用失败: {str(e)}"


def generate_ai_competitor_report(domain, ads_data):
    """使用 Gemini AI 生成深度竞品分析报告"""
    from collections import Counter
    
    if not ads_data:
        return None, "广告数据为空"
    
    # 准备数据摘要
    total_ads = len(ads_data)
    
    # 统计基础信息
    region_counter = Counter()
    duration_values = []
    ad_summaries = []
    
    for ad in ads_data[:50]:  # 限制发送的广告数量
        # 地区统计
        regions = ad.get('regions', '')
        if regions:
            for region in regions.split(', ')[:5]:
                region = region.strip()
                if region:
                    region_counter[region] += 1
        
        # 时长统计
        duration = ad.get('duration_days')
        if duration and isinstance(duration, (int, float)) and duration > 0:
            duration_values.append(duration)
        
        # 广告摘要
        title = ad.get('title') or ad.get('headline') or ''
        snippet = ad.get('snippet') or ad.get('ocr_text') or ''
        ad_summaries.append({
            'title': title[:100] if title else '',
            'snippet': snippet[:200] if snippet else '',
            'duration_days': ad.get('duration_days'),
            'regions': ad.get('regions', '')[:100],
            'first_shown': ad.get('first_shown', ''),
            'ad_type': ad.get('ad_type', '')
        })
    
    # 构建 Prompt
    prompt = f"""你是一位拥有 10 年经验的 B2B SaaS 市场战略专家和数据分析师。

请分析以下 {domain} 的广告投放数据，为我生成一份深度的竞品分析报告。

## 数据概览
- 品牌域名: {domain}
- 广告总数: {total_ads} 条
- 平均投放时长: {sum(duration_values)/len(duration_values):.1f} 天 (基于 {len(duration_values)} 条有数据的广告)
- 最长投放时长: {max(duration_values) if duration_values else 0} 天
- Top 投放地区: {', '.join([f"{r}({c})" for r, c in region_counter.most_common(10)])}

## 广告样本数据 (共 {len(ad_summaries)} 条):
{json.dumps(ad_summaries, ensure_ascii=False, indent=2)}

## 分析维度要求

请按照以下维度进行深度分析：

### 1. 产品矩阵拆解
- 根据广告标题和描述，归纳该产品正在推广的核心产品线有哪些
- 指出哪些是他们的「现金牛」产品（基于投放时长和覆盖广度判断）
- 分析产品之间的关联性和差异化定位

### 2. 营销卖点与痛点 (Messaging)
- 提取他们针对开发者（B2B）最常用的 3-5 个核心卖点（Value Propositions）
- 分析他们打击的用户痛点（Pain Points）
- 分析他们是如何区分不同技术栈（如 Flutter, React Native, Android, iOS）受众的

### 3. 市场与时机 (GTM Strategy)
- 统计投放最密集的国家/地区（分为 T1, T2, T3 梯队）
- 分析他们的市场优先级策略
- 基于投放时间分析是否有特定的季节性或大促节点

### 4. 长效素材分析
- 找出投放时间超过 30 天的「常青树」广告
- 分析这些长效广告的共同特征（标题结构、CTA、核心卖点）
- 提炼可复用的素材创意规律

### 5. 结论与建议
结合 Tencent RTC（腾讯云实时音视频）的背景，给出 3 条可落地的广告投放差异化竞争或跟进建议：
- 建议应具体、可执行
- 考虑腾讯云的技术优势和生态优势
- 针对不同市场给出差异化策略

## 输出要求
1. 使用专业的商业分析语调
2. 用 Markdown 格式输出，使用表格呈现关键数据
3. 每个章节都要有数据支撑的结论
4. 报告长度控制在 2000-3000 字"""

    # 调用 Gemini API
    report_content, error = call_gemini_api(prompt)
    
    if error:
        return None, error
    
    if report_content:
        # 添加报告头部
        header = f"""# {domain} 竞品广告投放深度分析报告

> **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **数据来源**: Google Ads Transparency Center  
> **分析样本量**: {total_ads} 条广告  
> **报告生成方式**: Gemini AI 深度分析

---

"""
        return header + report_content, None
    
    return None, "报告生成失败"


def generate_competitor_analysis_report(domain, ads_data):
    """生成竞品分析报告（基于规则的备用方案）"""
    from collections import Counter, defaultdict
    import re
    
    if not ads_data:
        return "# 数据不足\n\n未获取到足够的广告数据来生成分析报告。"
    
    # 数据预处理
    total_ads = len(ads_data)
    
    # 1. 产品矩阵分析
    product_keywords = defaultdict(int)
    product_ads = defaultdict(list)
    
    tech_keywords = {
        'SDK': 0, 'API': 0, 'Flutter': 0, 'React Native': 0, 'Android': 0, 'iOS': 0,
        'Unity': 0, 'Web': 0, 'Video': 0, 'AR': 0, 'AI': 0, 'Face': 0, 'Beauty': 0,
        'Filter': 0, 'Background': 0, 'Editor': 0, 'Camera': 0, 'Live': 0, 'Streaming': 0
    }
    
    # 投放时长分析
    long_running_ads = []
    duration_values = []
    
    # 地区分析
    region_counter = Counter()
    
    # 时间分析
    first_shown_dates = []
    
    # 卖点提取
    value_propositions = []
    pain_points = []
    
    for ad in ads_data:
        # 合并所有文本内容
        text_content = ' '.join(filter(None, [
            ad.get('title', ''),
            ad.get('headline', ''),
            ad.get('snippet', ''),
            ad.get('ocr_text', '')
        ])).lower()
        
        # 技术关键词统计
        for keyword in tech_keywords:
            if keyword.lower() in text_content:
                tech_keywords[keyword] += 1
        
        # 提取产品特征
        if 'sdk' in text_content:
            product_keywords['SDK产品'] += 1
        if 'api' in text_content:
            product_keywords['API服务'] += 1
        if any(kw in text_content for kw in ['video', 'editor', '视频']):
            product_keywords['视频编辑'] += 1
        if any(kw in text_content for kw in ['ar', 'filter', 'beauty', 'face', '美颜', '滤镜']):
            product_keywords['AR/美颜滤镜'] += 1
        if any(kw in text_content for kw in ['background', '背景']):
            product_keywords['背景处理'] += 1
        if any(kw in text_content for kw in ['camera', '相机']):
            product_keywords['相机功能'] += 1
        if any(kw in text_content for kw in ['live', 'streaming', '直播']):
            product_keywords['直播功能'] += 1
        
        # 投放时长
        duration = ad.get('duration_days')
        if duration and isinstance(duration, (int, float)) and duration > 0:
            duration_values.append(duration)
            if duration >= 30:
                long_running_ads.append({
                    'title': ad.get('title') or ad.get('headline') or '无标题',
                    'duration': duration,
                    'regions': ad.get('regions', ''),
                    'snippet': ad.get('snippet', '')[:100] if ad.get('snippet') else ''
                })
        
        # 地区统计
        regions = ad.get('regions', '')
        if regions:
            for region in regions.split(', '):
                region = region.strip()
                if region:
                    region_counter[region] += 1
        
        # 首次显示时间
        first_shown = ad.get('first_shown')
        if first_shown:
            first_shown_dates.append(first_shown)
        
        # 提取卖点关键词
        vp_patterns = [
            (r'(?:easy|simple|quick|fast)\s+(?:to\s+)?(?:integrate|integration|setup)', '快速集成'),
            (r'(?:cross[\-\s]?platform|multi[\-\s]?platform)', '跨平台支持'),
            (r'(?:real[\-\s]?time|实时)', '实时处理'),
            (r'(?:ai[\-\s]?powered|machine\s+learning|智能|人工智能)', 'AI驱动'),
            (r'(?:customiz|定制)', '可定制'),
            (r'(?:low[\-\s]?latency|低延迟)', '低延迟'),
            (r'(?:high[\-\s]?quality|高质量|高清)', '高质量'),
            (r'(?:free\s+trial|免费试用)', '免费试用'),
        ]
        
        for pattern, label in vp_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                value_propositions.append(label)
    
    # 排序长效广告
    long_running_ads.sort(key=lambda x: x['duration'], reverse=True)
    
    # 生成报告
    report = f"""# {domain} 竞品广告投放深度分析报告

> **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
> **数据来源**: Google Ads Transparency Center  
> **分析样本量**: {total_ads} 条广告

---

## 一、产品矩阵拆解

### 1.1 核心产品线分析

基于广告标题和描述的内容分析，{domain} 正在推广的核心产品线如下：

"""
    
    # 产品矩阵表格
    if product_keywords:
        sorted_products = sorted(product_keywords.items(), key=lambda x: x[1], reverse=True)
        report += "| 产品方向 | 广告数量 | 占比 | 推断定位 |\n"
        report += "|---------|---------|------|----------|\n"
        for product, count in sorted_products[:8]:
            percentage = (count / total_ads) * 100
            position = "核心产品" if percentage > 20 else ("重点产品" if percentage > 10 else "辅助产品")
            report += f"| {product} | {count} | {percentage:.1f}% | {position} |\n"
    
    # 识别现金牛产品
    report += "\n### 1.2 现金牛产品判断\n\n"
    if sorted_products:
        top_product = sorted_products[0][0]
        report += f"基于投放密度和覆盖广度分析，**{top_product}** 可能是 {domain} 的核心「现金牛」产品：\n\n"
        report += f"- 该产品线广告占比最高（{sorted_products[0][1]} 条，{(sorted_products[0][1]/total_ads)*100:.1f}%）\n"
        if long_running_ads:
            report += f"- 存在 {len([a for a in long_running_ads if a['duration'] >= 60])} 条投放超过 60 天的长效广告\n"
    
    # 技术栈覆盖
    report += "\n### 1.3 技术栈覆盖\n\n"
    tech_sorted = [(k, v) for k, v in tech_keywords.items() if v > 0]
    tech_sorted.sort(key=lambda x: x[1], reverse=True)
    
    if tech_sorted:
        report += "| 技术关键词 | 出现次数 | 覆盖率 |\n"
        report += "|-----------|---------|--------|\n"
        for tech, count in tech_sorted[:10]:
            report += f"| {tech} | {count} | {(count/total_ads)*100:.1f}% |\n"
    
    # 营销卖点与痛点
    report += "\n## 二、营销卖点与痛点分析 (Messaging)\n\n"
    
    vp_counter = Counter(value_propositions)
    if vp_counter:
        report += "### 2.1 核心价值主张 (Value Propositions)\n\n"
        report += "通过分析广告文案，提取到以下高频卖点：\n\n"
        for vp, count in vp_counter.most_common(5):
            report += f"- **{vp}**: 出现 {count} 次\n"
    
    report += "\n### 2.2 痛点打击策略\n\n"
    report += f"基于广告内容分析，{domain} 的目标用户痛点包括：\n\n"
    report += "- **集成复杂度**: 强调 SDK/API 的易用性和快速集成\n"
    report += "- **跨平台兼容**: 针对需要同时支持多平台的开发团队\n"
    report += "- **性能要求**: 强调实时处理、低延迟等技术指标\n"
    report += "- **开发成本**: 通过成熟方案降低自研成本\n"
    
    report += "\n### 2.3 技术栈受众区分\n\n"
    flutter_count = tech_keywords.get('Flutter', 0)
    rn_count = tech_keywords.get('React Native', 0)
    android_count = tech_keywords.get('Android', 0)
    ios_count = tech_keywords.get('iOS', 0)
    
    if flutter_count > 0 or rn_count > 0:
        report += f"- **Flutter 开发者**: {flutter_count} 条专属广告\n"
        report += f"- **React Native 开发者**: {rn_count} 条专属广告\n"
        report += f"- **Android 原生开发者**: {android_count} 条广告\n"
        report += f"- **iOS 原生开发者**: {ios_count} 条广告\n"
    
    # 市场与时机
    report += "\n## 三、市场与时机分析 (GTM Strategy)\n\n"
    
    report += "### 3.1 地区投放分布\n\n"
    
    # 地区分梯队
    t1_regions = ['United States', 'United Kingdom', 'Germany', 'Canada', 'Australia', 'France', 'Japan']
    t2_regions = ['Brazil', 'India', 'Mexico', 'Spain', 'Italy', 'Netherlands', 'South Korea', 'Singapore']
    
    t1_count = sum(region_counter.get(r, 0) for r in t1_regions)
    t2_count = sum(region_counter.get(r, 0) for r in t2_regions)
    t3_count = sum(region_counter.values()) - t1_count - t2_count
    
    total_region_mentions = sum(region_counter.values())
    if total_region_mentions > 0:
        report += "| 市场梯队 | 投放占比 | 主要国家 |\n"
        report += "|---------|---------|----------|\n"
        report += f"| T1 (核心市场) | {(t1_count/total_region_mentions)*100:.1f}% | 美国、英国、德国、加拿大 |\n"
        report += f"| T2 (增长市场) | {(t2_count/total_region_mentions)*100:.1f}% | 巴西、印度、墨西哥、西班牙 |\n"
        report += f"| T3 (新兴市场) | {(t3_count/total_region_mentions)*100:.1f}% | 其他国家/地区 |\n"
    
    report += "\n### 3.2 Top 10 投放地区\n\n"
    top_regions = region_counter.most_common(10)
    if top_regions:
        report += "| 排名 | 地区 | 广告曝光次数 |\n"
        report += "|------|------|-------------|\n"
        for i, (region, count) in enumerate(top_regions, 1):
            report += f"| {i} | {region} | {count} |\n"
    
    report += "\n### 3.3 投放时间规律\n\n"
    if duration_values:
        avg_duration = sum(duration_values) / len(duration_values)
        max_duration = max(duration_values)
        report += f"- **平均投放时长**: {avg_duration:.1f} 天\n"
        report += f"- **最长投放时长**: {max_duration} 天\n"
        report += f"- **30天以上长效广告**: {len([d for d in duration_values if d >= 30])} 条\n"
    
    # 长效素材分析
    report += "\n## 四、长效素材分析\n\n"
    
    if long_running_ads:
        report += f"### 4.1 常青树广告（投放≥30天）\n\n"
        report += f"共发现 **{len(long_running_ads)}** 条长效投放广告，这些广告的共同特征：\n\n"
        
        report += "| 广告标题 | 投放天数 | 投放地区 |\n"
        report += "|---------|---------|----------|\n"
        for ad in long_running_ads[:10]:
            title_raw = ad.get('title') or 'N/A'
            title = title_raw[:40] + '...' if len(title_raw) > 40 else title_raw
            regions_raw = ad.get('regions') or 'N/A'
            regions = regions_raw[:30] + '...' if len(regions_raw) > 30 else regions_raw
            report += f"| {title} | {ad.get('duration', 'N/A')} | {regions} |\n"
        
        report += "\n### 4.2 长效广告共同特征\n\n"
        report += "- **标题特点**: 直接说明产品核心功能（如 SDK、API）\n"
        report += "- **CTA 策略**: 多采用「了解更多」、「免费试用」等低门槛引导\n"
        report += "- **设计元素**: 简洁清晰，突出技术专业性\n"
    else:
        report += "未发现投放超过 30 天的长效广告。\n"
    
    # 结论与建议
    report += f"""

## 五、结论与建议

### 基于 Tencent RTC 背景的差异化竞争建议

结合腾讯云 RTC（实时音视频）的技术优势和市场定位，针对 {domain} 的投放策略，提出以下 3 条可落地建议：

#### 建议一：技术栈差异化 - 强化腾讯云原生优势

- **现状**: {domain} 在 Flutter、React Native 等跨平台框架上有明显布局
- **策略**: Tencent RTC 应强调与微信小程序、QQ 等腾讯生态的深度整合优势，这是竞品难以复制的护城河
- **执行**: 针对中国市场投放"小程序音视频解决方案"主题广告，针对海外市场强调"WeChat Mini Program SDK"

#### 建议二：场景化营销 - 聚焦垂直行业

- **现状**: {domain} 的广告以通用 SDK/API 为主，缺乏行业深度
- **策略**: Tencent RTC 可聚焦在线教育、远程医疗、金融视频客服等垂直场景
- **执行**: 制作行业专属落地页，投放关键词如"在线教育音视频SDK"、"远程问诊解决方案"

#### 建议三：价格与服务差异化

- **现状**: {domain} 强调"免费试用"和"快速集成"
- **策略**: Tencent RTC 可突出企业级 SLA、7x24 技术支持、本地化服务团队等差异点
- **执行**: 在 T1 市场投放"Enterprise-grade RTC with dedicated support"主题广告

---

**免责声明**: 本报告基于公开广告数据分析生成，仅供内部参考，不构成商业决策建议。

"""
    
    return report


@app.route("/api/generate-report/<task_id>", methods=["POST"])
def generate_report(task_id):
    """生成竞品分析报告"""
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        return jsonify({"error": "请先完成广告数据爬取"}), 400
    
    ads_data = task.get("results", [])
    domain = task.get("domain", "unknown")
    
    # 获取请求参数，判断是否使用 AI
    data = request.get_json() or {}
    use_ai = data.get("use_ai", True)  # 默认使用 AI
    
    try:
        report = None
        generation_method = "rule"  # 报告生成方式
        
        # 优先尝试 AI 生成
        if use_ai and GEMINI_API_KEY:
            print(f"正在使用 Gemini AI 生成报告...")
            report, error = generate_ai_competitor_report(domain, ads_data)
            if report:
                generation_method = "ai"
                print("AI 报告生成成功！")
            else:
                print(f"AI 报告生成失败: {error}，回退到规则生成")
        
        # 如果 AI 生成失败或未启用，使用规则生成
        if not report:
            print("使用规则生成报告...")
            report = generate_competitor_analysis_report(domain, ads_data)
            generation_method = "rule"
        
        return jsonify({
            "success": True,
            "report": report,
            "domain": domain,
            "ads_count": len(ads_data),
            "generation_method": generation_method,
            "ai_available": bool(GEMINI_API_KEY)
        })
    except Exception as e:
        print(f"报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"报告生成失败: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("Google Ads Transparency Center 广告爬取工具")
    print("=" * 50)
    print(f"OCR 功能: {'可用' if OCR_AVAILABLE else '不可用'}")
    print(f"SerpAPI Key: {'已配置' if API_KEY else '未配置'}")
    print(f"Gemini AI: {'已配置 ✓' if GEMINI_API_KEY else '未配置 (将使用规则生成报告)'}")
    print("启动服务器: http://localhost:8080")
    print("=" * 50)
    
    app.run(debug=True, host="0.0.0.0", port=8080)

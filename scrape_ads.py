"""
Google Ads Transparency Center 广告爬取脚本
使用 SerpAPI 来获取指定域名的文字广告标题和描述等信息
"""

from serpapi import GoogleSearch
import os
import json
from dotenv import load_dotenv
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from datetime import datetime
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("警告: PaddleOCR 未安装，无法识别图片中的文字。请运行: pip install paddlepaddle paddleocr")

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("SERPAPI_API_KEY")

if not API_KEY:
    raise ValueError("请设置环境变量 SERPAPI_API_KEY，或在 .env 文件中配置")

# 初始化 OCR（如果可用）
ocr_engine = None
if OCR_AVAILABLE:
    try:
        # 初始化 PaddleOCR，支持中英文
        # 注意：首次运行会下载模型文件，可能需要几分钟
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang='ch')  # 'ch' 支持中英文
        print("OCR 引擎初始化成功")
    except Exception as e:
        print(f"OCR 引擎初始化失败: {e}")
        ocr_engine = None


def get_ad_creatives_list(domain, region=None, creative_format="text", num=50):
    """
    第一步：获取指定域名的广告创意列表
    
    Args:
        domain: 域名，例如 "banuba.com"
        region: 区域，可选（如 "us", "uk" 等），None 表示不指定区域
        creative_format: 广告格式，默认 "text"（文字广告）
        num: 获取数量，默认 50
    
    Returns:
        广告创意列表
    """
    params_list = {
        "engine": "google_ads_transparency_center",
        "api_key": API_KEY,
        "text": domain,
        "creative_format": creative_format,
        "num": num
    }
    
    # 只有当 region 不为 None 时才添加 region 参数
    if region:
        params_list["region"] = region
    
    print(f"正在获取 {domain} 的广告创意列表...")
    search_list = GoogleSearch(params_list)
    results_list = search_list.get_dict()
    
    # 调试：打印返回的键名
    if "error" in results_list:
        print(f"API 错误: {results_list.get('error')}")
        return []
    
    # 尝试不同的可能字段名
    ads = results_list.get("ad_creatives", [])
    if not ads:
        ads = results_list.get("ads", [])
    if not ads:
        ads = results_list.get("organic_results", [])
    
    # 如果还是没有，打印所有键以便调试
    if not ads:
        print(f"API 返回的键: {list(results_list.keys())}")
        print(f"API 返回的完整数据（前500字符）: {str(results_list)[:500]}")
    
    print(f"找到 {len(ads)} 个广告创意")
    
    return ads


def extract_text_from_image(image_url):
    """
    从图片 URL 下载图片并使用 OCR 识别文字
    
    Args:
        image_url: 图片的 URL
    
    Returns:
        识别出的文字内容（字符串），如果失败返回 None
    """
    if not OCR_AVAILABLE or not ocr_engine:
        return None
    
    try:
        # 下载图片
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # 将图片内容转换为 numpy 数组
        image = Image.open(BytesIO(response.content))
        # 转换为 RGB 模式（如果不是的话）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)
        
        # 使用 PaddleOCR 识别（新版本使用 predict 方法）
        result = ocr_engine.predict(image_array)
        
        # 提取所有识别出的文字
        texts = []
        if result and isinstance(result, list) and len(result) > 0:
            # 新版本返回格式：列表包含字典，字典中有 'rec_texts' 和 'rec_scores'
            item = result[0]
            if isinstance(item, dict):
                rec_texts = item.get('rec_texts', [])
                rec_scores = item.get('rec_scores', [])
                
                # 组合文字和置信度
                for i, text in enumerate(rec_texts):
                    if text:
                        score = rec_scores[i] if i < len(rec_scores) else 1.0
                        if score > 0.5:  # 只保留置信度大于 0.5 的文字
                            texts.append(text)
        
        return "\n".join(texts) if texts else None
        
    except Exception as e:
        print(f"  OCR 识别失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_timestamp(timestamp):
    """
    将时间戳转换为可读的日期时间格式
    
    Args:
        timestamp: Unix 时间戳（整数）
    
    Returns:
        格式化的日期时间字符串，如果失败返回 None
    """
    if not timestamp:
        return None
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def format_date(date_int):
    """
    将日期整数（YYYYMMDD）转换为可读格式
    
    Args:
        date_int: 日期整数，如 20250118
    
    Returns:
        格式化的日期字符串，如果失败返回 None
    """
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
    """
    计算广告持续天数
    
    Args:
        first_shown: 首次显示时间戳或日期
        last_shown: 最后显示时间戳或日期
    
    Returns:
        持续天数（整数），如果无法计算返回 None
    """
    if not first_shown or not last_shown:
        return None
    try:
        # 如果是时间戳
        if isinstance(first_shown, (int, float)) and first_shown > 1000000000:
            first = datetime.fromtimestamp(first_shown)
            last = datetime.fromtimestamp(last_shown)
        # 如果是日期整数（YYYYMMDD）
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


def get_ad_details(advertiser_id, creative_id, region=None):
    """
    第二步：获取具体广告创意的详细信息（标题、描述等）
    
    Args:
        advertiser_id: 广告主ID
        creative_id: 创意ID
        region: 区域，可选（如 "us", "uk" 等），None 表示不指定区域
    
    Returns:
        (广告详细信息列表, 详情API的完整响应)
    """
    params_detail = {
        "engine": "google_ads_transparency_center_ad_details",
        "api_key": API_KEY,
        "advertiser_id": advertiser_id,
        "creative_id": creative_id
    }
    
    # 只有当 region 不为 None 时才添加 region 参数
    if region:
        params_detail["region"] = region
    
    search_detail = GoogleSearch(params_detail)
    res_detail = search_detail.get_dict()
    
    return res_detail.get("ad_creatives", []), res_detail


def save_to_excel(ads_data, excel_file):
    """
    将广告数据保存为 Excel 文件
    
    Args:
        ads_data: 广告数据列表
        excel_file: Excel 文件路径
    """
    if not ads_data:
        print("没有数据可保存到 Excel")
        return
    
    # 准备数据，将列表转换为字符串
    excel_data = []
    for ad in ads_data:
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
            "附加链接": ", ".join(ad.get("sitelink_texts", [])) if ad.get("sitelink_texts") else "",
            "附加链接描述": ", ".join(ad.get("sitelink_descriptions", [])) if ad.get("sitelink_descriptions") else "",
            "广告格式": ad.get("creative_format", ""),
            "广告类型": ad.get("ad_type", ""),
            "首次显示时间": ad.get("first_shown", ""),
            "最后显示时间": ad.get("last_shown", ""),
            "持续天数": ad.get("duration_days", ""),
            "发布地区": ad.get("regions", ""),
            "地区详情": ad.get("regions_detail", ""),
            "OCR识别文字": ad.get("ocr_text", ""),
            "备注": ad.get("note", ""),
            "详情链接": ad.get("details_link", ""),
        }
        excel_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(excel_data)
    
    # 保存为 Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"Excel 文件已保存到: {excel_file}")


def scrape_ads(domain="banuba.com", region="anywhere", output_file=None, excel_file=None):
    """
    爬取指定域名的文字广告信息
    
    Args:
        domain: 域名
        region: 区域
        output_file: 输出文件路径（可选，如果提供则保存为JSON）
        excel_file: Excel 文件路径（可选，如果提供则保存为Excel）
    
    Returns:
        包含所有广告详细信息的列表
    """
    # 第一步：获取广告创意列表
    ads_list = get_ad_creatives_list(domain, region)
    
    if not ads_list:
        print("未找到任何广告创意")
        return []
    
    # 第二步：获取每个广告的详细信息
    all_ads_details = []
    
    for idx, ad in enumerate(ads_list, 1):
        advertiser_id = ad.get("advertiser_id")
        creative_id = ad.get("ad_creative_id")
        
        if not advertiser_id or not creative_id:
            print(f"广告 {idx}: 缺少 advertiser_id 或 creative_id，跳过")
            continue
        
        print(f"正在获取广告 {idx}/{len(ads_list)} 的详细信息...")
        
        try:
            creatives, detail_response = get_ad_details(advertiser_id, creative_id, region)
            
            # 从详情 API 获取地区和时间信息
            search_info = detail_response.get("search_information", {})
            regions_info = search_info.get("regions", [])
            
            # 从列表 API 获取时间信息
            first_shown_ts = ad.get("first_shown")  # 时间戳
            last_shown_ts = ad.get("last_shown")  # 时间戳
            total_days_shown = ad.get("total_days_shown")  # 总天数
            
            # 格式化时间信息
            first_shown_formatted = format_timestamp(first_shown_ts) if first_shown_ts else None
            last_shown_formatted = format_timestamp(last_shown_ts) if last_shown_ts else None
            
            # 计算持续时间（如果列表API没有提供）
            duration_days = total_days_shown
            if not duration_days and first_shown_ts and last_shown_ts:
                duration_days = calculate_duration_days(first_shown_ts, last_shown_ts)
            
            # 格式化地区信息
            regions_list = []
            regions_detail = []
            if regions_info:
                for reg in regions_info:
                    region_name = reg.get("region_name", "")
                    first_shown_date = reg.get("first_shown")
                    last_shown_date = reg.get("last_shown")
                    times_shown = reg.get("times_shown", "")
                    
                    regions_list.append(region_name)
                    
                    region_detail = {
                        "region": region_name,
                        "first_shown": format_date(first_shown_date) if first_shown_date else None,
                        "last_shown": format_date(last_shown_date) if last_shown_date else None,
                        "times_shown": times_shown,
                        "duration_days": calculate_duration_days(first_shown_date, last_shown_date) if first_shown_date and last_shown_date else None
                    }
                    regions_detail.append(region_detail)
            
            # 如果详情 API 返回了 creatives，使用详情数据
            if creatives:
                for creative in creatives:
                    # 检查是否是图片广告（只有 image 字段）
                    if "image" in creative and not any(key in creative for key in ["title", "headline", "snippet", "long_headline"]):
                        # 这是图片广告，尝试使用 OCR 识别文字
                        image_url = creative.get("image")
                        ocr_text = None
                        
                        if image_url:
                            print(f"  正在使用 OCR 识别图片中的文字...")
                            ocr_text = extract_text_from_image(image_url)
                        
                        # 如果没有从详情 API 获取到图片，尝试从列表 API 获取
                        if not image_url and ad.get("image"):
                            image_url = ad.get("image")
                            print(f"  正在使用 OCR 识别图片中的文字（从列表API）...")
                            ocr_text = extract_text_from_image(image_url)
                        
                        # 解析 OCR 识别的文字（尝试提取标题和描述）
                        title = None
                        headline = None
                        snippet = None
                        
                        if ocr_text:
                            lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
                            if lines:
                                # 第一行作为标题
                                title = lines[0] if len(lines[0]) < 100 else None
                                # 第二行作为主标题
                                if len(lines) > 1:
                                    headline = lines[1] if len(lines[1]) < 200 else None
                                # 剩余行作为描述
                                if len(lines) > 2:
                                    snippet = " ".join(lines[2:])[:500]  # 限制长度
                                elif len(lines) == 1 and len(lines[0]) > 50:
                                    # 如果只有一行但很长，作为描述
                                    snippet = lines[0][:500]
                        
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
                            "sitelink_texts": [],
                            "sitelink_descriptions": [],
                            "creative_format": ad.get("format", "image"),  # 标记为图片
                            "ad_type": "图片广告",
                            "details_link": ad.get("details_link"),
                            "ocr_text": ocr_text,  # 保存完整的 OCR 识别结果
                            "note": "此广告为图片格式，已使用 OCR 识别文字" if ocr_text else "此广告为图片格式，OCR 识别失败或无文字",
                            # 时间信息
                            "first_shown": first_shown_formatted,
                            "last_shown": last_shown_formatted,
                            "duration_days": duration_days,
                            # 地区信息
                            "regions": ", ".join(regions_list) if regions_list else None,
                            "regions_detail": json.dumps(regions_detail, ensure_ascii=False) if regions_detail else None,
                        }
                    else:
                        # 文字广告，提取所有字段
                        ad_info = {
                            "advertiser_id": advertiser_id,
                            "creative_id": creative_id,
                            # 标题相关字段
                            "title": creative.get("title"),
                            "headline": creative.get("headline"),
                            "long_headline": creative.get("long_headline"),
                            # 描述相关字段
                            "snippet": creative.get("snippet"),
                            # 链接相关字段
                            "visible_link": creative.get("visible_link"),
                            "link": creative.get("link"),
                            "final_url": creative.get("final_url"),
                            # 其他信息
                            "call_to_action": creative.get("call_to_action"),
                            "sitelink_texts": creative.get("sitelink_texts", []),
                            "sitelink_descriptions": creative.get("sitelink_descriptions", []),
                            "creative_format": creative.get("creative_format") or ad.get("format"),
                            "ad_type": creative.get("ad_type"),
                            "details_link": ad.get("details_link"),
                            # 时间信息
                            "first_shown": first_shown_formatted,
                            "last_shown": last_shown_formatted,
                            "duration_days": duration_days,
                            # 地区信息
                            "regions": ", ".join(regions_list) if regions_list else None,
                            "regions_detail": json.dumps(regions_detail, ensure_ascii=False) if regions_detail else None,
                        }
                    
                    all_ads_details.append(ad_info)
                    
                    # 打印广告信息
                    print(f"\n广告 {idx}:")
                    if ad_info.get("note"):
                        print(f"  {ad_info['note']}")
                        if ad_info.get("ocr_text"):
                            print(f"  OCR 识别结果: {ad_info['ocr_text'][:200]}...")
                    else:
                        print(f"  标题 (Title): {ad_info['title']}")
                        if ad_info['headline']:
                            print(f"  标题 (Headline): {ad_info['headline']}")
                        if ad_info['long_headline']:
                            print(f"  长标题 (Long Headline): {ad_info['long_headline']}")
                        if ad_info['snippet']:
                            print(f"  描述 (Snippet): {ad_info['snippet']}")
                        if ad_info['call_to_action']:
                            print(f"  行动号召 (CTA): {ad_info['call_to_action']}")
                        print(f"  可见链接: {ad_info['visible_link']}")
                        if ad_info['link']:
                            print(f"  目标链接: {ad_info['link']}")
                        if ad_info['final_url']:
                            print(f"  最终URL: {ad_info['final_url']}")
                        if ad_info['sitelink_texts']:
                            print(f"  附加链接: {', '.join(ad_info['sitelink_texts'])}")
                    print("-" * 50)
            else:
                # 详情 API 没有返回 creatives，尝试从列表 API 的图片中识别文字
                image_url = ad.get("image")
                ocr_text = None
                
                if image_url:
                    print(f"  详情 API 未返回数据，尝试使用 OCR 识别列表中的图片...")
                    ocr_text = extract_text_from_image(image_url)
                
                # 解析 OCR 识别的文字
                title = None
                headline = None
                snippet = None
                
                if ocr_text:
                    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]
                    if lines:
                        title = lines[0] if len(lines[0]) < 100 else None
                        if len(lines) > 1:
                            headline = lines[1] if len(lines[1]) < 200 else None
                        if len(lines) > 2:
                            snippet = " ".join(lines[2:])[:500]
                        elif len(lines) == 1 and len(lines[0]) > 50:
                            snippet = lines[0][:500]
                
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
                    "sitelink_texts": [],
                    "sitelink_descriptions": [],
                    "creative_format": ad.get("format"),
                    "ad_type": None,
                    "details_link": ad.get("details_link"),
                    "ocr_text": ocr_text,
                    "note": "详情 API 未返回，已使用 OCR 识别图片文字" if ocr_text else "详情 API 未返回文字内容",
                    # 时间信息
                    "first_shown": first_shown_formatted,
                    "last_shown": last_shown_formatted,
                    "duration_days": duration_days,
                    # 地区信息
                    "regions": ", ".join(regions_list) if regions_list else None,
                    "regions_detail": json.dumps(regions_detail, ensure_ascii=False) if regions_detail else None,
                }
                all_ads_details.append(ad_info)
                if ocr_text:
                    print(f"\n广告 {idx}: OCR 识别成功")
                    if title:
                        print(f"  标题: {title}")
                    if headline:
                        print(f"  主标题: {headline}")
                    if snippet:
                        print(f"  描述: {snippet[:100]}...")
                else:
                    print(f"\n广告 {idx}: 详情 API 未返回文字内容")
                print("-" * 50)
        
        except Exception as e:
            print(f"获取广告 {idx} 详细信息时出错: {e}")
            continue
    
    # 保存到文件（如果指定）
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_ads_details, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果已保存到: {output_file}")
    
    # 保存为 Excel（如果指定）
    if excel_file:
        save_to_excel(all_ads_details, excel_file)
    
    return all_ads_details


if __name__ == "__main__":
    # 示例：爬取 banuba.com 的广告
    domain = "banuba.com"
    region = None  # 不指定区域，或使用有效的区域代码如 "us", "uk" 等
    
    print("=" * 60)
    print(f"开始爬取 {domain} 的文字广告信息")
    print("=" * 60)
    
    ads_data = scrape_ads(
        domain=domain,
        region=region,
        output_file="ads_results.json",
        excel_file="ads_results.xlsx"
    )
    
    print(f"\n总共获取到 {len(ads_data)} 个广告的详细信息")

# 🔍 AdInsight - Google Ads Transparency Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

一个强大的 Google Ads Transparency Center 广告数据分析工具，帮助你快速了解竞争对手的广告投放策略。

## ✨ 功能特性

### 📊 第一步：广告数据爬取
- 🌐 输入任意品牌域名，自动爬取其在 Google 投放的广告
- 🖼️ 支持文字广告和图片广告识别
- 🔤 **OCR 识别**：自动提取图片广告中的文字内容
- 📅 获取广告投放时间、持续天数、投放地区等详细信息
- 📄 支持导出 Excel 和 JSON 格式报告

### 🤖 第二步：AI 竞品分析报告
- 🧠 集成 **Gemini AI** 生成深度分析报告
- 📦 产品矩阵拆解：识别核心产品线和"现金牛"产品
- 💡 营销卖点与痛点分析
- 🌍 市场与时机 (GTM Strategy) 分析
- 🌲 长效素材分析：发现"常青树"广告特征
- 🎯 针对 Tencent RTC 的差异化竞争建议

### 🎨 用户界面
- 📱 响应式设计，支持 PC 和移动端
- 📖 分页展示所有广告（每页 9 个，3x3 网格）
- 🔍 点击广告卡片查看完整文案和投放地区
- 🌙 现代深色主题界面

## 🚀 快速开始

### 前置要求

- Python 3.9+
- [SerpAPI](https://serpapi.com/) API Key
- (可选) [Gemini API](https://apimart.ai/) Key（用于 AI 报告生成）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/YOUR_USERNAME/ad-insight.git
cd ad-insight
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**

创建 `.env` 文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Keys：
```env
SERPAPI_API_KEY=your_serpapi_key_here
GEMINI_API_KEY=your_gemini_key_here  # 可选
```

5. **启动服务**
```bash
python app.py
```

6. **访问应用**

打开浏览器访问 http://localhost:8080

## 📝 使用说明

1. 在输入框中输入要分析的品牌域名（如 `banuba.com`）
2. 选择要获取的广告数量
3. 勾选「启用 OCR」以识别图片广告中的文字
4. 点击「开始分析」等待爬取完成
5. 浏览广告列表，点击卡片查看详情
6. 下载 Excel/JSON 报告
7. （可选）点击「生成竞品分析报告」获取 AI 深度分析

## 🛠️ 技术栈

- **后端**: Python, Flask
- **前端**: HTML5, CSS3, JavaScript (原生)
- **API**: SerpAPI (Google Ads Transparency Center)
- **OCR**: PaddleOCR
- **AI**: Gemini Pro (via apimart.ai)

## 📁 项目结构

```
ad-insight/
├── app.py              # Flask 主应用
├── scrape_ads.py       # 原始爬取脚本（命令行版）
├── templates/
│   └── index.html      # 前端页面
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
├── .gitignore         
└── README.md
```

## ⚙️ 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `SERPAPI_API_KEY` | ✅ | SerpAPI 的 API Key |
| `GEMINI_API_KEY` | ❌ | Gemini AI 的 API Key（用于生成分析报告）|

## 🌐 部署

### 本地部署
```bash
python app.py
```

### Docker 部署
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "app.py"]
```

### 云平台部署

支持部署到以下平台：
- [Railway](https://railway.app/)
- [Render](https://render.com/)
- [Fly.io](https://fly.io/)

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请提交 Issue 或联系维护者。

---

<p align="center">
  Made with ❤️ for competitive intelligence
</p>

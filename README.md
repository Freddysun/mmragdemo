# 多模态RAG系统 (mmRAG)

基于Amazon Bedrock的多模态检索增强生成(RAG)系统，用于处理和分析包含文本、图片和表格的文档。

## 功能特点

- **多模态文档处理**：支持PDF、TXT、CSV、JPG、PNG等多种格式
- **图片识别与描述**：自动提取文档中的图片并使用LLM生成描述
- **表格识别与描述**：自动提取文档中的表格并使用LLM生成描述
- **文档分块与嵌入**：将处理后的文档分块并使用Bedrock的多模态嵌入模型生成嵌入
- **OpenSearch Serverless集成**：将嵌入存储到OpenSearch中用于高效检索
- **S3集成**：所有原始和处理后的文档、图片、表格和元数据都存储在S3中

## 系统架构

1. **文档解析器**：从S3读取文档，提取图片和表格，生成描述，处理后存回S3
2. **OpenSearch管理器**：创建和管理OpenSearch索引，执行向量搜索
3. **配置管理**：集中管理所有配置参数，包括AWS凭证、区域、模型ID等

## 安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/mmRAG.git
cd mmRAG
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置AWS凭证
```bash
aws configure
```

## 配置

所有配置参数都集中在`config.py`文件中，您可以直接编辑该文件来修改配置：

```python
# AWS区域配置
AWS_REGION = "us-west-2"  # 设置您的AWS区域

# S3配置
S3_BUCKET_NAME = "my-bucket-name"  # 设置您的S3存储桶名称

# OpenSearch Serverless配置
OPENSEARCH_COLLECTION_ENDPOINT = "my-collection-id.us-west-2.aoss.amazonaws.com"
OPENSEARCH_COLLECTION_ID = "my-collection-id"
OPENSEARCH_INDEX_NAME = "my-index-name"

# Bedrock知识库配置
BEDROCK_KB_ID = "my-kb-id"
BEDROCK_DS_ID = "my-ds-id"

# 文档处理配置
CHUNK_SIZE = 6000  # 文本分块大小
CHUNK_OVERLAP = 600  # 文本分块重叠大小
```

您也可以使用命令行更新配置：

```bash
python main.py update-config KEY VALUE
```

## 使用方法

### 处理文档

将文档上传到S3存储桶的`source/`目录，然后运行：

```bash
python main.py process
```

您也可以指定使用的LLM模型：

```bash
python main.py process --model anthropic.claude-3-sonnet-20240229-v1:0
```

### 设置OpenSearch索引

```bash
python main.py setup-opensearch
```

### 显示当前配置

```bash
python main.py show-config
```

## 文档处理流程

1. **PDF处理**：
   - 提取图片并使用LLM生成描述
   - 提取表格并使用LLM生成描述
   - 将图片和表格的描述回填到文档中
   - 将处理后的文档分块并嵌入到OpenSearch中

2. **文本文件处理**：
   - 直接分块并嵌入到OpenSearch中

3. **图片处理**：
   - 使用LLM生成描述
   - 将描述嵌入到OpenSearch中

## 目录结构

```
mmRAG/
├── config.py           # 全局配置参数
├── document_parser.py  # 文档解析器
├── opensearch_utils.py # OpenSearch工具
├── opensearch_schema.py # OpenSearch索引模式
├── main.py             # 主程序
├── requirements.txt    # 依赖列表
└── README.md           # 说明文档
```

## 依赖

- boto3: AWS SDK for Python
- langchain: LLM应用框架
- opensearch-py: OpenSearch Python客户端
- requests-aws4auth: AWS请求认证
- pymupdf: PDF处理
- pandas: 数据处理
- pillow: 图片处理

## 许可证

MIT

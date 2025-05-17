"""
文档解析器模块
用于处理来自S3的各种文档和图片，包括PDF、TXT、CSV、JPG、PNG等
处理步骤包括：
1. 提取图片并使用LLM生成描述
2. 提取表格并使用LLM生成描述
3. 将处理后的文档进行分块并嵌入到OpenSearch中
"""

import os
import io
import json
import uuid
import boto3
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import datetime
import base64
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# 导入配置
from config import (
    BedrockModels, DocumentProcessingConfig, 
    region_name, bucket_name, bedrock_client, s3_client,
    SOURCE_PREFIX, PROCESSED_PREFIX, IMAGES_PREFIX, TABLES_PREFIX, METADATA_PREFIX,
    TEMP_PATH, opensearch_config, get_config
)

# 导入OpenSearch索引模式
from opensearch_schema import get_index_mapping, create_document, create_simplified_document


class DocumentParser:
    """文档解析器类，用于处理各种类型的文档"""
    
    def __init__(self, model_id: str = BedrockModels.CLAUDE_SONNET):
        """初始化文档解析器"""
        self.model_id = model_id
        self.bedrock_client = bedrock_client
        self.s3_client = s3_client
        self.config = get_config()
        
        # 创建临时目录
        os.makedirs(TEMP_PATH, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=BedrockModels.TEXT_EMBEDDING
        )
        
        # 初始化OpenSearch客户端
        self.opensearch_client = self._init_opensearch_client()
        
        # 使用新的索引名称
        self.opensearch_index = "multimodal_index"
    
    def _init_opensearch_client(self) -> Optional[OpenSearch]:
        """初始化OpenSearch客户端"""
        if not opensearch_config.collection_endpoint:
            print("警告: OpenSearch Serverless集合端点未配置")
            return None
            
        service = opensearch_config.service
        region = opensearch_config.region
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token
        )
        
        client = OpenSearch(
            hosts=[{'host': opensearch_config.collection_endpoint, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )
        
        return client
    
    def _ensure_index_exists(self, client: Optional[OpenSearch] = None) -> bool:
        """确保OpenSearch索引存在"""
        if not client:
            client = self.opensearch_client
            
        if not client or not self.opensearch_index:
            return False
            
        if not client.indices.exists(index=self.opensearch_index):
            print(f"警告: 索引 {self.opensearch_index} 不存在，请先创建索引")
            return False
        return True
    
    def list_s3_files(self) -> List[Dict]:
        """列出S3存储桶中SOURCE_PREFIX目录下的所有文件"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=SOURCE_PREFIX
            )
            
            if 'Contents' not in response:
                return []
            
            files = []
            for item in response['Contents']:
                key = item['Key']
                # 跳过目录本身
                if key.endswith('/'):
                    continue
                
                # 获取文件扩展名
                _, ext = os.path.splitext(key.lower())
                
                # 检查是否为支持的文件类型
                if (ext in DocumentProcessingConfig.SUPPORTED_DOCUMENT_TYPES or 
                    ext in DocumentProcessingConfig.SUPPORTED_IMAGE_TYPES):
                    files.append({
                        'Key': key,
                        'Size': item['Size'],
                        'LastModified': item['LastModified'],
                        'Type': 'document' if ext in DocumentProcessingConfig.SUPPORTED_DOCUMENT_TYPES else 'image'
                    })
            
            return files
        
        except Exception as e:
            print(f"列出S3文件时出错: {str(e)}")
            return []
    
    def _download_file(self, key: str) -> str:
        """从S3下载文件到本地临时目录"""
        local_path = os.path.join(TEMP_PATH, os.path.basename(key))
        try:
            self.s3_client.download_file(bucket_name, key, local_path)
            return local_path
        except Exception as e:
            print(f"下载文件时出错: {str(e)}")
            return ""
    
    def _upload_file(self, local_path: str, s3_key: str) -> bool:
        """将本地文件上传到S3"""
        try:
            self.s3_client.upload_file(local_path, bucket_name, s3_key)
            return True
        except Exception as e:
            print(f"上传文件时出错: {str(e)}")
            return False
    
    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        """清理临时文件"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"清理临时文件时出错: {str(e)}")
    
    def _generate_description(self, prompt: str, image_base64: Optional[str] = None) -> str:
        """使用Bedrock模型生成描述"""
        try:
            if "anthropic.claude-3" in self.model_id.lower():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
                
                # 如果提供了图片，添加到消息中
                if image_base64:
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    })
                
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.2,
                    "messages": messages
                }
                
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(payload, ensure_ascii=False)
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
            else:
                print(f"不支持的模型类型: {self.model_id}")
                return ""
        except Exception as e:
            print(f"生成描述时出错: {str(e)}")
            return ""
    
    def _process_image(self, image_path: str, original_key: str, page_num: Optional[int] = None, 
                      img_index: Optional[int] = None) -> Dict:
        """处理图片并生成描述"""
        try:
            # 打开图片
            image = Image.open(image_path)
            
            # 检查图片尺寸是否符合最小要求
            if (image.width < DocumentProcessingConfig.IMAGE_MIN_WIDTH or 
                image.height < DocumentProcessingConfig.IMAGE_MIN_HEIGHT):
                return {}
            
            # 检查图片是否为空白或几乎为空白（仅当提供了图片路径时）
            img_array = np.array(image)
            if img_array.ndim == 3:  # 彩色图片
                # 计算非白色像素的比例
                non_white_ratio = np.sum(np.any(img_array < 240, axis=2)) / (image.width * image.height)
                if non_white_ratio < 0.05:  # 如果非白色像素少于5%，认为是空白图片
                    return {}
            else:  # 灰度图片
                non_white_ratio = np.sum(img_array < 240) / (image.width * image.height)
                if non_white_ratio < 0.05:
                    return {}
            
            # 将图片转换为Base64
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # 使用LLM生成图片描述
            description = self._generate_description(
                prompt="请详细描述这张图片的内容，包括图片中的主要对象、场景、文字内容等。",
                image_base64=image_base64
            )
            
            # 生成唯一的图片ID
            image_id = str(uuid.uuid4())
            image_filename = f"{image_id}{os.path.splitext(image_path)[1]}"
            
            # 构建图片元数据
            image_metadata = {
                "id": image_id,
                "original_file": original_key,
                "width": image.width,
                "height": image.height,
                "description": description,
                "s3_path": f"{IMAGES_PREFIX}{image_filename}"
            }
            
            # 添加页码和图片索引（如果提供）
            if page_num is not None:
                image_metadata["page_number"] = page_num
            if img_index is not None:
                image_metadata["image_index"] = img_index
            
            # 上传图片到S3
            self._upload_file(image_path, f"{IMAGES_PREFIX}{image_filename}")
            
            # 保存元数据
            metadata_path = os.path.join(TEMP_PATH, f"{image_id}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(image_metadata, f, indent=2, ensure_ascii=False)
            
            # 上传元数据到S3
            self._upload_file(
                metadata_path, 
                f"{METADATA_PREFIX}images/{image_id}_metadata.json"
            )
            
            # 清理临时文件
            self._cleanup_temp_files([metadata_path])
            
            return image_metadata
            
        except Exception as e:
            print(f"处理图片时出错: {str(e)}")
            return {}
    
    def _extract_images_from_pdf(self, pdf_path: str, original_key: str) -> List[Dict]:
        """从PDF文件中提取图片，并使用LLM生成描述"""
        images_info = []
        processed_images = set()  # 用于跟踪已处理的图片哈希值
        temp_files = []
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            
            # 遍历每一页
            for page_num, page in enumerate(pdf_document):
                # 获取页面上的图片
                image_list = page.get_images(full=True)
                
                # 遍历页面上的每个图片
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # 图片的xref
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 计算图片哈希值以避免重复
                    image_hash = hash(image_bytes)
                    if image_hash in processed_images:
                        continue
                    
                    # 添加到已处理集合
                    processed_images.add(image_hash)
                    
                    # 生成临时图片文件
                    image_id = str(uuid.uuid4())
                    image_filename = f"{image_id}.jpg"
                    local_image_path = os.path.join(TEMP_PATH, image_filename)
                    temp_files.append(local_image_path)
                    
                    # 将图片字节转换为PIL图像并保存
                    image = Image.open(io.BytesIO(image_bytes))
                    image.save(local_image_path, "JPEG")
                    
                    # 处理图片
                    image_metadata = self._process_image(
                        local_image_path, 
                        original_key, 
                        page_num + 1, 
                        img_index + 1
                    )
                    
                    if image_metadata:
                        images_info.append(image_metadata)
            
            # 关闭PDF文档
            pdf_document.close()
            
        except Exception as e:
            print(f"提取PDF图片时出错: {str(e)}")
        finally:
            # 清理临时文件
            self._cleanup_temp_files(temp_files)
            
        return images_info
    
    def _extract_tables_from_pdf(self, pdf_path: str, original_key: str) -> List[Dict]:
        """从PDF文件中提取表格，并使用LLM生成描述"""
        tables_info = []
        temp_files = []
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            
            # 遍历每一页
            for page_num, page in enumerate(pdf_document):
                # 尝试提取页面上的表格
                try:
                    tables = page.find_tables()
                    
                    # 遍历页面上的每个表格
                    for table_index, table in enumerate(tables):
                        # 检查表格尺寸是否符合最小要求
                        if (len(table.cells) < DocumentProcessingConfig.TABLE_MIN_ROWS or 
                            len(table.cells[0]) < DocumentProcessingConfig.TABLE_MIN_COLS):
                            continue
                        
                        # 提取表格数据
                        table_data = []
                        for row in range(len(table.cells)):
                            row_data = []
                            for col in range(len(table.cells[0])):
                                try:
                                    cell = table.cells[row][col]
                                    if cell:
                                        row_data.append(cell.text)
                                    else:
                                        row_data.append("")
                                except IndexError:
                                    row_data.append("")
                            table_data.append(row_data)
                        
                        # 生成唯一的表格ID
                        table_id = str(uuid.uuid4())
                        
                        # 将表格数据转换为CSV格式
                        csv_data = io.StringIO()
                        pd.DataFrame(table_data).to_csv(csv_data, index=False, header=False)
                        csv_content = csv_data.getvalue()
                        
                        # 保存CSV到本地
                        table_filename = f"{table_id}.csv"
                        local_table_path = os.path.join(TEMP_PATH, table_filename)
                        temp_files.append(local_table_path)
                        
                        with open(local_table_path, 'w') as f:
                            f.write(csv_content)
                        
                        # 使用LLM生成表格描述
                        description = self._generate_description(
                            prompt=f"请详细描述这个表格的内容和结构。表格数据如下:\n\n{csv_content}"
                        )
                        
                        # 构建表格元数据
                        table_metadata = {
                            "id": table_id,
                            "original_file": original_key,
                            "page_number": page_num + 1,
                            "table_index": table_index + 1,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "description": description,
                            "s3_path": f"{TABLES_PREFIX}{table_filename}"
                        }
                        
                        # 上传表格到S3
                        self._upload_file(local_table_path, f"{TABLES_PREFIX}{table_filename}")
                        
                        # 保存元数据
                        metadata_path = os.path.join(TEMP_PATH, f"{table_id}_metadata.json")
                        temp_files.append(metadata_path)
                        
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(table_metadata, f, indent=2, ensure_ascii=False)
                        
                        # 上传元数据到S3
                        self._upload_file(
                            metadata_path, 
                            f"{METADATA_PREFIX}tables/{table_id}_metadata.json"
                        )
                        
                        # 添加到结果列表
                        tables_info.append(table_metadata)
                        
                except Exception as e:
                    print(f"提取页面 {page_num+1} 的表格时出错: {str(e)}")
                    continue
            
            # 关闭PDF文档
            pdf_document.close()
            
        except Exception as e:
            print(f"提取PDF表格时出错: {str(e)}")
        finally:
            # 清理临时文件
            self._cleanup_temp_files(temp_files)
            
        return tables_info
    
    def _index_chunks_to_opensearch(self, chunks: List[str], source_key: str) -> bool:
        """将文本分块嵌入到OpenSearch中"""
        if not self.opensearch_client or not self.opensearch_index:
            return False
        
        try:
            # 确保索引存在
            if not self._ensure_index_exists():
                return False
            
            # 为每个分块生成嵌入并索引
            for i, chunk in enumerate(chunks):
                try:
                    # 生成嵌入
                    embedding = self.embedding_model.embed_query(chunk)
                    
                    # 构建文档
                    chunk_id = f"{os.path.basename(source_key)}_{i}"
                    
                    # 创建文档结构，将嵌入向量放入text_embedding字段
                    doc = {
                        "chunk_id": chunk_id,
                        "content": chunk,
                        "document_id": os.path.basename(source_key),
                        "document_type": os.path.splitext(source_key)[1][1:].lower(),
                        "source": source_key,
                        "metadata": {
                            "filename": os.path.basename(source_key),
                            "filepath": os.path.dirname(source_key),
                            "chunk_index": i,
                            "created_at": datetime.datetime.now().isoformat(),
                            "file_type": os.path.splitext(source_key)[1][1:].lower()
                        },
                        "text_embedding": embedding
                    }
                    
                    # 索引文档
                    self.opensearch_client.index(
                        index=self.opensearch_index,
                        body=doc
                    )
                    print(f"成功索引文档: {chunk_id} 到 {self.opensearch_index}")
                except Exception as chunk_error:
                    print(f"索引单个分块时出错: {str(chunk_error)}")
                    continue
            
            return True
        
        except Exception as e:
            print(f"索引到OpenSearch时出错: {str(e)}")
            return False
    
    def _split_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DocumentProcessingConfig.CHUNK_SIZE,
            chunk_overlap=DocumentProcessingConfig.CHUNK_OVERLAP
        )
        return text_splitter.split_text(text)
    
    def process_pdf(self, file_key: str) -> Dict:
        """处理PDF文件"""
        result = {
            "original_key": file_key,
            "processed_key": "",
            "images": [],
            "tables": [],
            "chunks": 0,
            "status": "failed"
        }
        
        temp_files = []
        
        try:
            # 下载PDF文件
            local_path = self._download_file(file_key)
            if not local_path:
                return result
                
            temp_files.append(local_path)
            
            # 提取图片
            images_info = self._extract_images_from_pdf(local_path, file_key)
            result["images"] = images_info
            
            # 提取表格
            tables_info = self._extract_tables_from_pdf(local_path, file_key)
            result["tables"] = tables_info
            
            # 打开PDF文件
            pdf_document = fitz.open(local_path)
            
            # 提取所有文本并添加图片和表格描述，使用Markdown格式
            text_content = ""
            
            # 遍历每一页
            for page_num, page in enumerate(pdf_document):
                # 获取页面纯文本，保留段落结构
                page_text = page.get_text("text")
                
                # 添加到总文本内容
                text_content += page_text + "\n\n"
                
                # 添加图片描述，使用<figure>标签
                for img_info in images_info:
                    if img_info.get("page_number") == page_num + 1:
                        # 在文本中添加图片描述
                        text_content += f"\n<figure>\n"
                        text_content += f"![{img_info['description']}](s3://{bucket_name}/{img_info['s3_path']})\n"
                        text_content += f"</figure>\n\n"
                
                # 添加表格描述
                for table_info in tables_info:
                    if table_info["page_number"] == page_num + 1:
                        # 在文本中添加表格描述
                        text_content += f"\n<table>\n"
                        text_content += f"<caption>{table_info['description']}</caption>\n"
                        text_content += f"<tr><td>表格数据位置: s3://{bucket_name}/{table_info['s3_path']}</td></tr>\n"
                        text_content += f"</table>\n\n"
            
            # 创建一个新的文本文件，用于存储处理后的内容
            processed_filename = os.path.basename(file_key).replace('.pdf', '.md')
            processed_local_path = os.path.join(TEMP_PATH, f"processed_{processed_filename}")
            temp_files.append(processed_local_path)
            
            # 保存为Markdown文件
            with open(processed_local_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
                
            # 上传处理后的文件到S3
            processed_key = f"{PROCESSED_PREFIX}{processed_filename}"
            self._upload_file(processed_local_path, processed_key)
            result["processed_key"] = processed_key
            
            # 关闭PDF文档
            pdf_document.close()
            
            # 使用RecursiveCharacterTextSplitter进行分块
            chunks = self._split_text(text_content)
            result["chunks"] = len(chunks)
            
            # 如果OpenSearch客户端已初始化，则将分块嵌入到OpenSearch中
            if self.opensearch_client and self.opensearch_index:
                self._index_chunks_to_opensearch(chunks, file_key)
            
            result["status"] = "success"
            return result
            
        except Exception as e:
            print(f"处理PDF文件时出错: {str(e)}")
            return result
        finally:
            # 清理临时文件
            self._cleanup_temp_files(temp_files)
    
    def process_text_file(self, file_key: str) -> Dict:
        """处理文本文件（TXT、CSV）"""
        result = {
            "original_key": file_key,
            "processed_key": file_key,  # 对于文本文件，不创建新的处理文件
            "images": [],
            "tables": [],
            "chunks": 0,
            "status": "failed"
        }
        
        try:
            # 下载文本文件
            local_path = self._download_file(file_key)
            if not local_path:
                return result
            
            # 读取文本内容
            with open(local_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # 使用RecursiveCharacterTextSplitter进行分块
            chunks = self._split_text(text_content)
            result["chunks"] = len(chunks)
            
            # 如果OpenSearch客户端已初始化，则将分块嵌入到OpenSearch中
            if self.opensearch_client and self.opensearch_index:
                self._index_chunks_to_opensearch(chunks, file_key)
            
            result["status"] = "success"
            return result
        
        except Exception as e:
            print(f"处理文本文件时出错: {str(e)}")
            return result
        finally:
            # 清理临时文件
            if 'local_path' in locals() and local_path:
                self._cleanup_temp_files([local_path])
    
    def process_image_file(self, file_key: str) -> Dict:
        """处理图片文件（JPG、PNG）"""
        result = {
            "original_key": file_key,
            "processed_key": "",
            "images": [],
            "tables": [],
            "chunks": 0,
            "status": "failed"
        }
        
        try:
            # 下载图片文件
            local_path = self._download_file(file_key)
            if not local_path:
                return result
            
            # 处理图片
            image_metadata = self._process_image(local_path, file_key)
            
            if not image_metadata:
                return result
                
            # 添加到结果列表
            result["images"].append(image_metadata)
            result["processed_key"] = image_metadata["s3_path"]
            
            # 如果OpenSearch客户端已初始化，则将图片描述嵌入到OpenSearch中
            if self.opensearch_client and self.opensearch_index:
                self._index_chunks_to_opensearch([image_metadata["description"]], file_key)
                result["chunks"] = 1
            
            result["status"] = "success"
            return result
        
        except Exception as e:
            print(f"处理图片文件时出错: {str(e)}")
            return result
        finally:
            # 清理临时文件
            if 'local_path' in locals() and local_path:
                self._cleanup_temp_files([local_path])
    
    def process_file(self, file_key: str) -> Dict:
        """处理单个文件"""
        _, ext = os.path.splitext(file_key.lower())
        
        if ext == '.pdf':
            return self.process_pdf(file_key)
        elif ext in ['.txt', '.csv']:
            return self.process_text_file(file_key)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self.process_image_file(file_key)
        else:
            return {
                "original_key": file_key,
                "status": "unsupported_format"
            }
    
    def process_all_files(self) -> List[Dict]:
        """处理S3存储桶中所有文件"""
        results = []
        
        # 列出S3文件
        files = self.list_s3_files()
        
        # 处理每个文件
        for file_info in files:
            result = self.process_file(file_info['Key'])
            results.append(result)
        
        return results


# 如果直接运行此脚本，则处理所有文件
if __name__ == "__main__":
    parser = DocumentParser()
    results = parser.process_all_files()
    
    # 打印处理结果
    print(json.dumps(results, indent=2, ensure_ascii=False))

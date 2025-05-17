#!/usr/bin/env python3
"""
处理图片元数据文件，提取描述信息，进行embedding，并存储到OpenSearch索引中
支持纯文本embedding和多模态(图片+文本)embedding
"""

import os
import sys
import json
import boto3
import time
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加mmRAG目录到Python路径
sys.path.append('/home/ec2-user/mmRAG')

from config import (
    bucket_name, METADATA_PREFIX, IMAGES_PREFIX,
    BedrockModels, region_name, DEFAULT_MULTIMODAL_EMBEDDING_MODEL
)
from opensearch_utils import OpenSearchManager
from opensearch_schema import create_document

# 初始化客户端
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
opensearch_manager = OpenSearchManager()

def get_multimodal_embedding(text: str, image_s3_path: str, model_id: str = DEFAULT_MULTIMODAL_EMBEDDING_MODEL) -> List[float]:
    """
    使用Bedrock获取图片和文本的多模态embedding向量
    
    Args:
        text: 要进行embedding的文本
        image_s3_path: 图片在S3中的路径
        model_id: 使用的多模态embedding模型ID
        
    Returns:
        多模态embedding向量
    """
    try:
        # 从S3下载图片
        print(f"正在从S3下载图片: {image_s3_path}")
        image_response = s3_client.get_object(
            Bucket=bucket_name,
            Key=image_s3_path
        )
        image_data = image_response['Body'].read()
        
        # 将图片转换为base64编码
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 准备请求体
        request_body = {
            "inputText": text,
            "inputImage": base64_image
        }
        
        # 调用Bedrock API
        print(f"正在调用多模态embedding模型: {model_id}")
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # 解析响应
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
        
        if not embedding:
            print("警告: 响应中没有找到embedding字段")
            return []
            
        # 打印embedding维度信息
        print(f"多模态embedding维度: {len(embedding)}")
        
        return embedding
    
    except Exception as e:
        print(f"获取多模态embedding时出错: {str(e)}")
        return []

def list_image_metadata_files() -> List[str]:
    """
    列出S3存储桶中的所有图片元数据文件
    
    Returns:
        元数据文件路径列表
    """
    print(f"正在列出S3存储桶 {bucket_name} 中的图片元数据文件...")
    
    metadata_prefix = f"{METADATA_PREFIX}images/"
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=metadata_prefix
    )
    
    metadata_files = []
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('_metadata.json'):
                metadata_files.append(obj['Key'])
    
    print(f"找到 {len(metadata_files)} 个图片元数据文件")
    return metadata_files

def get_embedding(text: str, model_id: str = BedrockModels.TEXT_EMBEDDING) -> List[float]:
    """
    使用Bedrock获取文本的embedding向量
    
    Args:
        text: 要进行embedding的文本
        model_id: 使用的embedding模型ID
        
    Returns:
        embedding向量
    """
    try:
        # 准备请求体
        request_body = {
            "texts": [text],
            "input_type": "search_document"
        }
        
        # 调用Bedrock API
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # 解析响应
        response_body = json.loads(response['body'].read())
        embedding = response_body['embeddings'][0]
        
        # 确保embedding是正确的维度
        if len(embedding) != 1024:
            print(f"警告: embedding维度为 {len(embedding)}，预期为1024")
        
        return embedding
    
    except Exception as e:
        print(f"获取embedding时出错: {str(e)}")
        return []

def process_metadata_file(metadata_key: str) -> bool:
    """
    处理单个元数据文件
    
    Args:
        metadata_key: 元数据文件的S3键
        
    Returns:
        处理是否成功
    """
    try:
        # 从S3下载元数据文件
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=metadata_key
        )
        
        # 解析元数据
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        # 提取描述信息
        image_id = metadata.get('id')
        description = metadata.get('description', '')
        original_file = metadata.get('original_file', '')
        s3_path = metadata.get('s3_path', '')
        
        if not description or not image_id:
            print(f"元数据文件 {metadata_key} 缺少必要信息，跳过")
            return False
        
        # 获取描述的embedding
        print(f"正在为图片 {image_id} 的描述生成文本embedding...")
        text_embedding = get_embedding(description)
        
        if not text_embedding:
            print(f"无法为图片 {image_id} 的描述生成文本embedding，跳过")
            return False
        
        # 打印embedding的前10个值和长度，用于调试
        print(f"文本Embedding长度: {len(text_embedding)}")
        print(f"文本Embedding前10个值: {text_embedding[:10]}")
        
        # 获取多模态embedding (图片+文本)
        print(f"正在为图片 {image_id} 生成多模态embedding...")
        multimodal_embedding = get_multimodal_embedding(description, s3_path)
        
        if not multimodal_embedding:
            print(f"无法为图片 {image_id} 生成多模态embedding，但将继续处理文本embedding")
        else:
            print(f"多模态Embedding长度: {len(multimodal_embedding)}")
            print(f"多模态Embedding前10个值: {multimodal_embedding[:10]}")
        
        # 准备文档
        source = original_file
        chunk_id = f"img_{image_id}"
        document_type = "image"
        
        # 准备元数据
        doc_metadata = {
            "filename": os.path.basename(original_file) if original_file else "",
            "filepath": os.path.dirname(original_file) if original_file else "",
            "created_at": datetime.now().isoformat(),
            "file_type": "image",
            "image_info": {
                "width": metadata.get('width', 0),
                "height": metadata.get('height', 0),
                "s3_path": s3_path
            }
        }
        
        # 获取索引名称
        index_name = 'multimodal_index'
        
        # 创建文档 - 注意字段名称必须与索引映射匹配
        document = {
            "content": description,
            "source": source,
            "chunk_id": chunk_id,
            "document_type": document_type,
            "document_id": image_id,
            "text_embedding": text_embedding,  # 使用text_embedding字段存储文本描述的向量
            "metadata": doc_metadata
        }
        
        # 如果有多模态embedding，添加到文档中
        if multimodal_embedding:
            document["multimodal_embedding"] = multimodal_embedding
        
        # 使用OpenSearch客户端直接索引文档，不指定ID
        print(f"正在将图片 {image_id} 的描述存储到OpenSearch...")
        response = opensearch_manager.opensearch_client.index(
            index=index_name,
            body=document
        )
        
        if response.get('result') == 'created':
            print(f"成功将图片 {image_id} 的描述存储到OpenSearch")
            return True
        else:
            print(f"存储图片 {image_id} 的描述到OpenSearch失败: {response}")
            return False
    
    except Exception as e:
        print(f"处理元数据文件 {metadata_key} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    # 列出所有元数据文件
    metadata_files = list_image_metadata_files()
    
    if not metadata_files:
        print("没有找到图片元数据文件，退出")
        return
    
    # 处理每个元数据文件
    success_count = 0
    for i, metadata_file in enumerate(metadata_files):
        print(f"正在处理 {i+1}/{len(metadata_files)}: {metadata_file}")
        if process_metadata_file(metadata_file):
            success_count += 1
        
        # 避免API限制，每处理10个文件暂停1秒
        if (i + 1) % 10 == 0:
            time.sleep(1)
    
    print(f"处理完成: 成功 {success_count}/{len(metadata_files)}")

if __name__ == "__main__":
    main()

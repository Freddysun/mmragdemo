#!/usr/bin/env python3
"""
测试图片描述的搜索功能
"""

import os
import sys
import json
import boto3
from typing import Dict, List, Any

# 添加mmRAG目录到Python路径
sys.path.append('/home/ec2-user/mmRAG')

from config import bucket_name, region_name
from opensearch_utils import OpenSearchManager

# 初始化OpenSearch管理器
opensearch_manager = OpenSearchManager()

def search_by_text(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    使用文本查询搜索图片描述
    
    Args:
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    # 构建查询
    search_body = {
        "size": k,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "content": query
                        }
                    },
                    {
                        "term": {
                            "document_type": "image"
                        }
                    }
                ]
            }
        }
    }
    
    # 执行搜索
    index_name = opensearch_manager.opensearch_client.indices.get('*').popitem()[0]
    response = opensearch_manager.opensearch_client.search(
        index=index_name,
        body=search_body
    )
    
    # 处理结果
    results = []
    for hit in response['hits']['hits']:
        result = {
            'score': hit['_score'],
            'content': hit['_source'].get('content', ''),
            'document_id': hit['_source'].get('document_id', ''),
            'source': hit['_source'].get('source', ''),
            'metadata': hit['_source'].get('metadata', {})
        }
        results.append(result)
    
    return results

def main():
    """主函数"""
    # 测试查询
    queries = [
        "VPC对等连接",
        "AWS架构图",
        "网络拓扑",
        "安全组",
        "子网"
    ]
    
    for query in queries:
        print(f"\n\n查询: '{query}'")
        print("-" * 50)
        
        results = search_by_text(query, k=3)
        
        if not results:
            print(f"没有找到与 '{query}' 相关的图片")
            continue
        
        for i, result in enumerate(results):
            print(f"\n结果 {i+1} (得分: {result['score']:.2f}):")
            print(f"图片ID: {result['document_id']}")
            print(f"来源: {result['source']}")
            
            # 获取图片路径
            image_info = result.get('metadata', {}).get('image_info', {})
            s3_path = image_info.get('s3_path', '')
            if s3_path:
                print(f"图片路径: s3://{bucket_name}/{s3_path}")
            
            # 打印描述的前200个字符
            content = result.get('content', '')
            if content:
                print(f"描述: {content[:200]}..." if len(content) > 200 else f"描述: {content}")

if __name__ == "__main__":
    main()

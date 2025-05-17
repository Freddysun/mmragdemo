"""
OpenSearch工具模块
用于创建和管理OpenSearch索引、查询和管理文档
"""

import json
import boto3
from typing import Dict, List, Any, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# 导入配置
from config import opensearch_config, region_name
from opensearch_schema import get_index_mapping, create_document, create_simplified_document

class OpenSearchManager:
    """OpenSearch管理器类，用于管理OpenSearch索引和文档"""
    
    def __init__(self):
        """初始化OpenSearch管理器"""
        self.opensearch_client = self._init_opensearch_client()
    
    def _init_opensearch_client(self) -> Optional[OpenSearch]:
        """
        初始化OpenSearch客户端
        
        Returns:
            OpenSearch客户端实例，如果配置不完整则返回None
        """
        if not opensearch_config.collection_endpoint:
            print("警告: OpenSearch Serverless集合端点未配置，无法初始化OpenSearch客户端")
            return None
        
        try:
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
            
            # 确保endpoint不包含https://前缀
            endpoint = opensearch_config.collection_endpoint
            if endpoint.startswith("https://"):
                endpoint = endpoint.replace("https://", "")
            
            client = OpenSearch(
                hosts=[{'host': endpoint, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
            
            return client
        
        except Exception as e:
            print(f"初始化OpenSearch客户端时出错: {str(e)}")
            return None
    
    def create_index(self, index_name: str, dimension: int = 1536) -> bool:
        """
        创建OpenSearch索引
        
        Args:
            index_name: 索引名称
            dimension: 向量维度
            
        Returns:
            创建是否成功
        """
        if not self.opensearch_client:
            return False
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 检查索引是否已存在
            if self.opensearch_client.indices.exists(index=index_name):
                print(f"索引 {index_name} 已存在")
                return True
            
            # 创建索引
            index_mapping = get_index_mapping(use_simplified=True)
            
            self.opensearch_client.indices.create(
                index=index_name,
                body=index_mapping
            )
            
            print(f"成功创建索引 {index_name}")
            return True
        
        except Exception as e:
            print(f"创建索引时出错: {str(e)}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        """
        删除OpenSearch索引
        
        Args:
            index_name: 索引名称
            
        Returns:
            删除是否成功
        """
        if not self.opensearch_client:
            return False
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 检查索引是否存在
            if not self.opensearch_client.indices.exists(index=index_name):
                print(f"索引 {index_name} 不存在")
                return True
            
            # 删除索引
            self.opensearch_client.indices.delete(index=index_name)
            
            print(f"成功删除索引 {index_name}")
            return True
        
        except Exception as e:
            print(f"删除索引时出错: {str(e)}")
            return False
    
    def index_document(self, index_name: str, doc_id: str, document: Dict) -> bool:
        """
        索引单个文档
        
        Args:
            index_name: 索引名称
            doc_id: 文档ID
            document: 文档内容
            
        Returns:
            索引是否成功
        """
        if not self.opensearch_client:
            return False
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 确保索引存在
            if not self.opensearch_client.indices.exists(index=index_name):
                print(f"索引 {index_name} 不存在，正在创建...")
                if not self.create_index(index_name):
                    return False
            
            # 索引文档
            self.opensearch_client.index(
                index=index_name,
                body=document,
                id=doc_id
            )
            
            return True
        
        except Exception as e:
            print(f"索引文档时出错: {str(e)}")
            return False
    
    def bulk_index_documents(self, index_name: str, documents: List[Dict]) -> bool:
        """
        批量索引文档
        
        Args:
            index_name: 索引名称
            documents: 文档列表，每个文档应包含'id'字段
            
        Returns:
            索引是否成功
        """
        if not self.opensearch_client:
            return False
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 确保索引存在
            if not self.opensearch_client.indices.exists(index=index_name):
                print(f"索引 {index_name} 不存在，正在创建...")
                if not self.create_index(index_name):
                    return False
            
            # 准备批量索引请求
            bulk_body = []
            for doc in documents:
                doc_id = doc.pop('id', None)
                if not doc_id:
                    continue
                
                # 添加索引操作
                bulk_body.append({"index": {"_index": index_name, "_id": doc_id}})
                # 添加文档内容
                bulk_body.append(doc)
            
            # 执行批量索引
            if bulk_body:
                self.opensearch_client.bulk(body=bulk_body)
            
            return True
        
        except Exception as e:
            print(f"批量索引文档时出错: {str(e)}")
            return False
    
    def search_by_vector(self, index_name: str, vector: List[float], k: int = 5) -> List[Dict]:
        """
        使用向量搜索文档
        
        Args:
            index_name: 索引名称
            vector: 查询向量
            k: 返回的最大结果数
            
        Returns:
            搜索结果列表
        """
        if not self.opensearch_client:
            return []
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 构建查询
            query = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": vector,
                            "k": k
                        }
                    }
                }
            }
            
            # 执行搜索
            response = self.opensearch_client.search(
                body=query,
                index=index_name
            )
            
            # 解析结果
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"向量搜索时出错: {str(e)}")
            return []
    
    def search_by_text(self, index_name: str, text: str, k: int = 5) -> List[Dict]:
        """
        使用文本搜索文档
        
        Args:
            index_name: 索引名称
            text: 查询文本
            k: 返回的最大结果数
            
        Returns:
            搜索结果列表
        """
        if not self.opensearch_client:
            return []
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 构建查询
            query = {
                "size": k,
                "query": {
                    "match": {
                        "content": text
                    }
                }
            }
            
            # 执行搜索
            response = self.opensearch_client.search(
                body=query,
                index=index_name
            )
            
            # 解析结果
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"文本搜索时出错: {str(e)}")
            return []
    
    def hybrid_search(self, index_name: str, text: str, vector: List[float], k: int = 5) -> List[Dict]:
        """
        混合搜索（文本 + 向量）
        
        Args:
            index_name: 索引名称
            text: 查询文本
            vector: 查询向量
            k: 返回的最大结果数
            
        Returns:
            搜索结果列表
        """
        if not self.opensearch_client:
            return []
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 构建查询
            query = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": text,
                                        "boost": 0.3  # 文本搜索权重
                                    }
                                }
                            },
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": vector,
                                        "k": k,
                                        "boost": 0.7  # 向量搜索权重
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            
            # 执行搜索
            response = self.opensearch_client.search(
                body=query,
                index=index_name
            )
            
            # 解析结果
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"混合搜索时出错: {str(e)}")
            return []
    
    def get_index_stats(self, index_name: str) -> Dict:
        """
        获取索引统计信息
        
        Args:
            index_name: 索引名称
            
        Returns:
            索引统计信息
        """
        if not self.opensearch_client:
            return {}
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 获取索引统计信息
            stats = self.opensearch_client.indices.stats(index=index_name)
            
            # 提取关键信息
            result = {
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "store_size": stats["_all"]["primaries"]["store"]["size_in_bytes"],
                "index_name": index_name
            }
            
            return result
        
        except Exception as e:
            print(f"获取索引统计信息时出错: {str(e)}")
            return {}



    def search_by_vector_with_filter(self, index_name: str, vector: List[float], filter_condition: Dict = None, k: int = 5) -> List[Dict]:
        """
        使用向量搜索文档，支持过滤条件
        
        Args:
            index_name: 索引名称
            vector: 查询向量
            filter_condition: 过滤条件
            k: 返回的最大结果数
            
        Returns:
            搜索结果列表
        """
        if not self.opensearch_client:
            return []
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            # 构建查询
            if filter_condition:
                query = {
                    "size": k,
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "knn": {
                                        "embedding": {
                                            "vector": vector,
                                            "k": k
                                        }
                                    }
                                }
                            ],
                            "filter": filter_condition
                        }
                    }
                }
            else:
                query = {
                    "size": k,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": k
                            }
                        }
                    }
                }
            
            # 执行搜索
            response = self.opensearch_client.search(
                body=query,
                index=index_name
            )
            
            # 解析结果
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    **hit["_source"]
                }
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"向量搜索时出错: {str(e)}")
            return []
    
    def get_all_sources(self, index_name: str) -> List[str]:
        """
        获取索引中的所有文档来源
        
        Args:
            index_name: 索引名称
            
        Returns:
            文档来源列表
        """
        if not self.opensearch_client:
            return []
        
        try:
            # 强制使用 multimodal_index 作为索引名称
            index_name = 'multimodal_index'
            
            # 构建聚合查询
            query = {
                "size": 0,
                "aggs": {
                    "sources": {
                        "terms": {
                            "field": "source",
                            "size": 1000
                        }
                    }
                }
            }
            
            # 执行搜索
            response = self.opensearch_client.search(
                body=query,
                index=index_name
            )
            
            # 解析结果
            sources = []
            for bucket in response.get('aggregations', {}).get('sources', {}).get('buckets', []):
                sources.append(bucket.get('key'))
            
            print(f"索引 {index_name} 中的所有来源: {sources}")
            return sources
        
        except Exception as e:
            print(f"获取文档来源时出错: {str(e)}")
            return []
    

    def rerank_results(self, query: str, results: List[Dict], model_id: str) -> List[Dict]:
        """
        对搜索结果进行重排序
        
        Args:
            query: 查询文本
            results: 搜索结果列表
            model_id: 重排序模型ID
            
        Returns:
            重排序后的结果列表
        """
        try:
            # 如果结果为空，直接返回
            if not results:
                return []
            
            # 创建Bedrock客户端
            import boto3
            from config import region_name
            bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
            
            # 准备请求体
            passages = []
            for result in results:
                passages.append({
                    "id": result.get("id", ""),
                    "text": result.get("content", "")
                })
            
            request_body = {
                "query": query,
                "passages": passages
            }
            
            # 调用Bedrock API
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # 解析响应
            response_body = json.loads(response['body'].read())
            reranked_passages = response_body.get("passages", [])
            
            # 创建ID到分数的映射
            id_to_score = {passage["id"]: passage["score"] for passage in reranked_passages}
            
            # 更新结果分数
            for result in results:
                result_id = result.get("id", "")
                if result_id in id_to_score:
                    result["score"] = id_to_score[result_id]
            
            # 按分数排序
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return results
        
        except Exception as e:
            print(f"重排序结果时出错: {str(e)}")
            return results
    
# 如果直接运行此脚本，则创建示例索引
if __name__ == "__main__":
    manager = OpenSearchManager()
    
    # 创建示例索引
    index_name = 'multimodal_index'
    success = manager.create_index(index_name)
    index_name = opensearch_config.index or "mmrag-test-index"
    success = manager.create_index(index_name)
    
    if success:
        print(f"成功创建索引: {index_name}")
    else:
        print(f"创建索引失败: {index_name}")

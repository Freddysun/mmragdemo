"""
OpenSearch索引模式设计
基于Amazon Bedrock Knowledge Base的索引结构设计
"""

import json
from typing import Dict, Any, List, Optional

# 简化版索引映射，适用于OpenSearch Serverless
SIMPLIFIED_INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            # 文档内容
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            # 文档标题
            "title": {
                "type": "text",
                "analyzer": "standard"
            },
            # 文档来源
            "source": {
                "type": "keyword"
            },
            # 文档类型
            "document_type": {
                "type": "keyword"
            },
            # 文档ID
            "document_id": {
                "type": "keyword"
            },
            # 分块ID
            "chunk_id": {
                "type": "keyword"
            },
            # 文本嵌入向量
            "text_embedding": {
                "type": "knn_vector",
                "dimension": 1536
            },
            # 图像嵌入向量（如果有）
            "image_embedding": {
                "type": "knn_vector",
                "dimension": 1024
            },
            # 多模态嵌入向量
            "multimodal_embedding": {
                "type": "knn_vector",
                "dimension": 1536
            },
            # 元数据
            "metadata": {
                "type": "object",
                "enabled": True
            }
        }
    }
}

# 基于Amazon Bedrock Knowledge Base的索引映射
BEDROCK_KB_INDEX_MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 5,
            "number_of_replicas": 1,
            "knn": True,
            "knn.algo_param.ef_search": 512
        }
    },
    "mappings": {
        "properties": {
            # 文档内容
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            # 文档标题
            "title": {
                "type": "text",
                "analyzer": "standard"
            },
            # 文档来源
            "source": {
                "type": "keyword"
            },
            # 文档类型
            "document_type": {
                "type": "keyword"
            },
            # 文档ID
            "document_id": {
                "type": "keyword"
            },
            # 分块ID
            "chunk_id": {
                "type": "keyword"
            },
            # 文本嵌入向量
            "text_embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            # 图像嵌入向量（如果有）
            "image_embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            # 多模态嵌入向量
            "multimodal_embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            # 元数据
            "metadata": {
                "type": "object",
                "properties": {
                    # 原始文件名
                    "filename": {
                        "type": "keyword"
                    },
                    # 原始文件路径
                    "filepath": {
                        "type": "keyword"
                    },
                    # 页码（对于PDF）
                    "page_number": {
                        "type": "integer"
                    },
                    # 创建时间
                    "created_at": {
                        "type": "date"
                    },
                    # 更新时间
                    "updated_at": {
                        "type": "date"
                    },
                    # 文件类型
                    "file_type": {
                        "type": "keyword"
                    },
                    # 处理状态
                    "processing_status": {
                        "type": "keyword"
                    },
                    # 图片信息（如果是图片）
                    "image_info": {
                        "type": "object",
                        "properties": {
                            "width": {
                                "type": "integer"
                            },
                            "height": {
                                "type": "integer"
                            },
                            "format": {
                                "type": "keyword"
                            },
                            "s3_path": {
                                "type": "keyword"
                            }
                        }
                    },
                    # 表格信息（如果是表格）
                    "table_info": {
                        "type": "object",
                        "properties": {
                            "rows": {
                                "type": "integer"
                            },
                            "columns": {
                                "type": "integer"
                            },
                            "s3_path": {
                                "type": "keyword"
                            }
                        }
                    }
                }
            }
        }
    }
}

def get_index_mapping(use_simplified: bool = False) -> Dict[str, Any]:
    """
    获取索引映射
    
    Args:
        use_simplified: 是否使用简化版索引映射
        
    Returns:
        索引映射字典
    """
    if use_simplified:
        return SIMPLIFIED_INDEX_MAPPING
    else:
        return BEDROCK_KB_INDEX_MAPPING

def create_document(
    content: str,
    source: str,
    chunk_id: str,
    embedding: List[float],
    title: Optional[str] = None,
    document_type: Optional[str] = None,
    document_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    image_embedding: Optional[List[float]] = None,
    multimodal_embedding: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    创建文档对象
    
    Args:
        content: 文档内容
        source: 文档来源
        chunk_id: 分块ID
        embedding: 文本嵌入向量
        title: 文档标题
        document_type: 文档类型
        document_id: 文档ID
        metadata: 元数据
        image_embedding: 图像嵌入向量
        multimodal_embedding: 多模态嵌入向量
        
    Returns:
        文档对象字典
    """
    doc = {
        "content": content,
        "source": source,
        "chunk_id": chunk_id,
        "text_embedding": embedding
    }
    
    if title:
        doc["title"] = title
    
    if document_type:
        doc["document_type"] = document_type
    
    if document_id:
        doc["document_id"] = document_id
    
    if metadata:
        doc["metadata"] = metadata
    
    if image_embedding:
        doc["image_embedding"] = image_embedding
    
    if multimodal_embedding:
        doc["multimodal_embedding"] = multimodal_embedding
    
    return doc

def create_simplified_document(
    content: str,
    source: str,
    chunk_id: str,
    embedding: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建简化版文档对象
    
    Args:
        content: 文档内容
        source: 文档来源
        chunk_id: 分块ID
        embedding: 嵌入向量
        metadata: 元数据
        
    Returns:
        文档对象字典
    """
    doc = {
        "content": content,
        "source": source,
        "chunk_id": chunk_id,
        "text_embedding": embedding  # 修改为统一使用text_embedding字段
    }
    
    if metadata:
        doc["metadata"] = metadata
    
    return doc

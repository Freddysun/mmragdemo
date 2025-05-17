"""
多模态RAG系统主程序
用于处理文档、图片和表格，并提供检索增强生成功能
"""

import os
import argparse
import json
from typing import Dict, List, Any, Optional

from document_parser import DocumentParser
from opensearch_utils import OpenSearchManager
from config import (
    BedrockModels, DocumentProcessingConfig, 
    bucket_name, SOURCE_PREFIX, PROCESSED_PREFIX,
    opensearch_config, kb_config, get_config, update_config, print_config
)

def process_documents(model_id: str = BedrockModels.DEFAULT_LLM) -> List[Dict]:
    """
    处理S3存储桶中的所有文档
    
    Args:
        model_id: 用于图片和表格描述的LLM模型ID
        
    Returns:
        处理结果信息列表
    """
    parser = DocumentParser(model_id=model_id)
    results = parser.process_all_files()
    return results

def setup_opensearch_index() -> bool:
    """
    设置OpenSearch索引
    
    Returns:
        设置是否成功
    """
    manager = OpenSearchManager()
    
    # 使用配置中的索引名称
    index_to_use = opensearch_config.index
    
    if not index_to_use:
        print("错误: 配置中没有设置索引名称")
        return False
    
    # 创建索引
    success = manager.create_index(index_to_use)
    
    if success:
        print(f"成功创建索引: {index_to_use}")
    else:
        print(f"创建索引失败: {index_to_use}")
    
    return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态RAG系统")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 处理文档子命令
    process_parser = subparsers.add_parser("process", help="处理S3存储桶中的文档")
    process_parser.add_argument(
        "--model", 
        default=BedrockModels.DEFAULT_LLM,
        help="用于图片和表格描述的LLM模型ID"
    )
    
    # 设置OpenSearch索引子命令
    subparsers.add_parser("setup-opensearch", help="设置OpenSearch索引")
    
    # 显示配置子命令
    subparsers.add_parser("show-config", help="显示当前配置")
    
    # 更新配置子命令
    update_config_parser = subparsers.add_parser("update-config", help="更新配置")
    update_config_parser.add_argument("key", help="配置键")
    update_config_parser.add_argument("value", help="配置值")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行相应的命令
    if args.command == "process":
        results = process_documents(model_id=args.model)
        print(json.dumps(results, indent=2))
    elif args.command == "setup-opensearch":
        setup_opensearch_index()
    elif args.command == "show-config":
        print_config()
    elif args.command == "update-config":
        update_config(args.key, args.value)
        print(f"已更新配置: {args.key} = {args.value}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

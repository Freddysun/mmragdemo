"""
用户权限管理模块
用于从DynamoDB获取用户权限
"""

import boto3
from typing import List, Dict, Any, Optional

def get_user_permissions(username: str) -> List[str]:
    """
    从DynamoDB获取用户权限
    
    Args:
        username: 用户名
        
    Returns:
        用户有权访问的文档列表
    """
    try:
        # 初始化DynamoDB客户端，指定区域为us-west-2
        dynamodb = boto3.client('dynamodb', region_name='us-west-2')
        
        # 扫描整个表
        response = dynamodb.scan(TableName='content_permission')
        items = response.get('Items', [])
        
        # 处理分页结果
        while 'LastEvaluatedKey' in response:
            response = dynamodb.scan(
                TableName='content_permission',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        
        # 打印调试信息
        print(f"DynamoDB表中的所有项目: {items}")
        
        # 过滤出用户有权访问的文档
        docs = []
        for item in items:
            if 'users' in item and 'doc' in item:
                # 获取文档ID
                doc_id = item['doc']['S']
                
                # 检查用户是否在users列表中
                users_list = item['users']['L']
                for user_item in users_list:
                    if 'S' in user_item and user_item['S'] == username:
                        docs.append(doc_id)
                        break
        
        # 如果没有找到任何项目，返回空列表
        if not docs:
            print(f"用户 {username} 没有任何权限")
            return []
        
        # 检查是否有通配符权限
        if '*' in docs:
            print(f"用户 {username} 有权访问所有文档")
            return ["*"]
        
        print(f"用户 {username} 有权访问以下文档: {docs}")
        return docs
    
    except Exception as e:
        import traceback
        print(f"获取用户权限时出错: {str(e)}")
        print(traceback.format_exc())
        # 出错时返回空列表，表示没有权限
        return []

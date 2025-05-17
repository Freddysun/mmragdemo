"""
权限管理工具模块
用于管理文档访问权限和用户认证
"""

import boto3
import json
from typing import Dict, List, Any, Optional
from botocore.exceptions import ClientError

class PermissionManager:
    """权限管理类，用于管理文档访问权限"""
    
    def __init__(self, table_name: str = "content_permission", region: str = "us-west-2"):
        """
        初始化权限管理器
        
        Args:
            table_name: DynamoDB表名
            region: AWS区域
        """
        self.table_name = table_name
        self.region = region
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = None
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> bool:
        """
        确保DynamoDB表存在，如果不存在则创建
        
        Returns:
            表是否存在或创建成功
        """
        try:
            # 检查表是否存在
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']
            
            if self.table_name in existing_tables:
                self.table = self.dynamodb.Table(self.table_name)
                return True
            
            # 创建表
            self.table = self.dynamodb.create_table(
                TableName=self.table_name,
                KeySchema=[
                    {
                        'AttributeName': 'doc',
                        'KeyType': 'HASH'  # 分区键
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'doc',
                        'AttributeType': 'S'
                    }
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            )
            
            # 等待表创建完成
            self.table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
            print(f"表 {self.table_name} 创建成功")
            return True
            
        except Exception as e:
            print(f"确保表存在时出错: {str(e)}")
            return False
    
    def add_document_permission(self, doc_id: str, users: List[str]) -> bool:
        """
        添加文档访问权限
        
        Args:
            doc_id: 文档ID
            users: 用户列表
            
        Returns:
            操作是否成功
        """
        if not self.table:
            return False
        
        try:
            response = self.table.put_item(
                Item={
                    'doc': doc_id,
                    'users': users
                }
            )
            return True
        except Exception as e:
            print(f"添加文档权限时出错: {str(e)}")
            return False
    
    def get_user_documents(self, username: str) -> List[str]:
        """
        获取用户有权访问的文档列表
        
        Args:
            username: 用户名
            
        Returns:
            文档ID列表
        """
        if not self.table:
            return []
        
        try:
            # 获取所有文档
            response = self.table.scan()
            items = response.get('Items', [])
            
            # 处理分页结果
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # 过滤出用户有权访问的文档
            docs = []
            for item in items:
                if 'users' in item and 'doc' in item:
                    # 提取用户列表
                    user_list = [user_item.get('S') for user_item in item['users'].get('L', [])]
                    
                    # 检查用户是否在列表中
                    if username in user_list:
                        docs.append(item['doc'])
            
            # 打印找到的文档
            print(f"用户 {username} 有权访问的文档: {docs}")
            
            return docs
        
        except Exception as e:
            print(f"获取用户文档时出错: {str(e)}")
            return []
    
    def update_document_permission(self, doc_id: str, users: List[str]) -> bool:
        """
        更新文档访问权限
        
        Args:
            doc_id: 文档ID
            users: 用户列表
            
        Returns:
            操作是否成功
        """
        return self.add_document_permission(doc_id, users)
    
    def remove_document_permission(self, doc_id: str) -> bool:
        """
        删除文档访问权限
        
        Args:
            doc_id: 文档ID
            
        Returns:
            操作是否成功
        """
        if not self.table:
            return False
        
        try:
            response = self.table.delete_item(
                Key={
                    'doc': doc_id
                }
            )
            return True
        except Exception as e:
            print(f"删除文档权限时出错: {str(e)}")
            return False


class CognitoAuthenticator:
    """Cognito认证类，用于用户认证"""
    
    def __init__(self, user_pool_id: str = "us-west-2_SeqJ15BmT", client_id: str = None, region: str = "us-west-2"):
        """
        初始化Cognito认证器
        
        Args:
            user_pool_id: Cognito用户池ID
            client_id: 应用客户端ID
            region: AWS区域
        """
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.region = region
        self.cognito_idp = boto3.client('cognito-idp', region_name=region)
        
        # 如果没有提供client_id，尝试获取第一个客户端ID
        if not self.client_id:
            try:
                response = self.cognito_idp.list_user_pool_clients(
                    UserPoolId=self.user_pool_id,
                    MaxResults=1
                )
                if response['UserPoolClients']:
                    self.client_id = response['UserPoolClients'][0]['ClientId']
                    print(f"使用客户端ID: {self.client_id}")
            except Exception as e:
                print(f"获取客户端ID时出错: {str(e)}")
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        认证用户
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            认证结果，包含token和用户信息
        """
        if not self.client_id:
            return {"error": "未配置客户端ID"}
        
        # 简化认证过程，仅检查用户名是否存在于Cognito用户池中
        try:
            print(f"尝试认证用户: {username}")
            
            # 检查用户是否存在
            try:
                user_info = self.cognito_idp.admin_get_user(
                    UserPoolId=self.user_pool_id,
                    Username=username
                )
                
                # 简化认证，不验证密码（仅用于演示）
                # 在实际生产环境中，应该使用proper的密码验证
                print(f"用户 {username} 存在，跳过密码验证")
                
                # 模拟认证成功
                return {
                    "success": True,
                    "id_token": "mock-id-token",
                    "access_token": "mock-access-token",
                    "refresh_token": "mock-refresh-token",
                    "expires_in": 3600,
                    "username": username,
                    "attributes": {
                        "email": next((attr["Value"] for attr in user_info.get("UserAttributes", []) if attr["Name"] == "email"), f"{username}@example.com")
                    }
                }
                
            except self.cognito_idp.exceptions.UserNotFoundException:
                print(f"用户 {username} 不存在")
                return {"error": "用户名或密码错误"}
                
        except Exception as e:
            print(f"认证时出错: {str(e)}")
            return {"error": f"认证时出错: {str(e)}"}
    
    def _get_user_attributes(self, username: str, access_token: str = None) -> Dict[str, str]:
        """
        获取用户属性
        
        Args:
            username: 用户名
            access_token: 访问令牌
            
        Returns:
            用户属性字典
        """
        try:
            if access_token:
                # 使用访问令牌获取用户信息
                response = self.cognito_idp.get_user(
                    AccessToken=access_token
                )
                
                # 转换属性列表为字典
                attributes = {}
                for attr in response.get('UserAttributes', []):
                    attributes[attr['Name']] = attr['Value']
                
                return attributes
            else:
                # 使用管理API获取用户信息
                response = self.cognito_idp.admin_get_user(
                    UserPoolId=self.user_pool_id,
                    Username=username
                )
                
                # 转换属性列表为字典
                attributes = {}
                for attr in response.get('UserAttributes', []):
                    attributes[attr['Name']] = attr['Value']
                
                return attributes
        
        except Exception as e:
            print(f"获取用户属性时出错: {str(e)}")
            return {}
    
    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        刷新令牌
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            刷新结果，包含新的token
        """
        if not self.client_id:
            return {"error": "未配置客户端ID"}
        
        try:
            response = self.cognito_idp.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='REFRESH_TOKEN_AUTH',
                AuthParameters={
                    'REFRESH_TOKEN': refresh_token
                }
            )
            
            # 提取认证结果
            auth_result = response.get('AuthenticationResult', {})
            
            return {
                "success": True,
                "id_token": auth_result.get('IdToken'),
                "access_token": auth_result.get('AccessToken'),
                "expires_in": auth_result.get('ExpiresIn')
            }
        
        except Exception as e:
            return {"error": f"刷新令牌时出错: {str(e)}"}

#!/usr/bin/env python3
"""
多模态RAG系统的Streamlit前端界面 - 带权限管理
"""

import os
import sys
import json
import boto3
import streamlit as st
import tempfile
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple

# 添加mmRAG目录到Python路径
sys.path.append('/home/ec2-user/mmRAG')

from config import bucket_name, region_name, BedrockModels, opensearch_config
from get_user_permissions import get_user_permissions
from opensearch_utils import OpenSearchManager
from permission_utils import PermissionManager, CognitoAuthenticator

# 导入combined_search模块中的函数
from combined_search import generate_answer

# 初始化客户端
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
opensearch_manager = OpenSearchManager()
permission_manager = PermissionManager()
cognito_auth = CognitoAuthenticator()

# 设置页面配置
st.set_page_config(
    page_title="多模态RAG系统",
    page_icon="🔍",
    layout="wide"
)

# 初始化会话状态
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'refresh_token' not in st.session_state:
    st.session_state.refresh_token = None
if 'authorized_docs' not in st.session_state:
    st.session_state.authorized_docs = []

def get_image_from_s3(s3_path: str) -> Optional[Tuple[Image.Image, bytes]]:
    """
    从S3获取图片
    
    Args:
        s3_path: S3路径
        
    Returns:
        PIL图片对象和原始字节数据的元组，如果获取失败则返回None
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
        image_data = response['Body'].read()
        image = Image.open(BytesIO(image_data))
        return image, image_data
    except Exception as e:
        st.error(f"获取图片时出错: {str(e)}")
        return None

def combined_search(query: str, text_k: int = 5, image_k: int = 3, use_rerank: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    组合搜索文本和图片
    
    Args:
        query: 查询文本
        text_k: 文本结果数量
        image_k: 图片结果数量
        use_rerank: 是否使用rerank
        
    Returns:
        文本结果和图片结果的元组
    """
    # 获取用户有权访问的文档列表
    authorized_docs = st.session_state.authorized_docs
    
    # 如果没有授权文档，返回空列表
    if not authorized_docs:
        st.warning("您没有权限访问任何文档")
        return [], []
    
    try:
        # 获取所有可用的文档来源 - 使用硬编码的索引名称
        index_name = 'multimodal_index'
        available_sources = opensearch_manager.get_all_sources(index_name)
        
        # 打印调试信息
        print(f"可用的文档来源: {available_sources}")
        print(f"用户授权的文档: {authorized_docs}")
        
        # 如果authorized_docs是通配符"*"，则使用所有可用的文档来源
        if authorized_docs == ["*"]:
            filter_sources = available_sources
            st.info(f"您有权访问所有文档 ({len(available_sources)} 个文档来源)")
        else:
            # 否则，使用authorized_docs和available_sources的交集
            # 这里使用部分匹配，如果source中包含authorized_docs中的任何一个，则认为有权访问
            filter_sources = []
            for source in available_sources:
                for doc in authorized_docs:
                    # 如果doc是字符串，检查source是否包含doc
                    if isinstance(doc, str):
                        if doc in source:
                            filter_sources.append(source)
                            print(f"匹配成功: 文档 {doc} 在来源 {source} 中找到")
                            break
                    # 如果doc是字典，提取S字段
                    elif isinstance(doc, dict) and 'S' in doc:
                        doc_str = doc['S']
                        if doc_str in source:
                            filter_sources.append(source)
                            print(f"匹配成功: 文档 {doc_str} 在来源 {source} 中找到")
                            break
            
            st.info(f"您有权访问 {len(filter_sources)} 个文档来源 (共 {len(available_sources)} 个)")
        
        if not filter_sources:
            st.warning("没有找到匹配的文档来源")
            return [], []
        
        # 准备过滤条件 - 使用source字段
        filter_condition = {
            "terms": {
                "source": filter_sources
            }
        }
        
        # 使用json.dumps确保使用双引号
        filter_json = json.dumps(filter_condition)
        print(f"搜索过滤条件: {filter_json}")
        
        # 导入combined_search模块中的函数
        from combined_search import combined_search as cs
        
        # 调用combined_search.py中的combined_search函数，传入过滤条件
        return cs(query, text_k=text_k, image_k=image_k, use_rerank=use_rerank, filter_condition=json.loads(filter_json))
    
    except Exception as e:
        import traceback
        print(f"搜索时出错: {str(e)}")
        print(traceback.format_exc())
        st.error(f"搜索时出错: {str(e)}")
        return [], []
        return [], []

def display_text_result(result: Dict[str, Any], index: int):
    """
    显示文本搜索结果
    
    Args:
        result: 搜索结果
        index: 结果索引
    """
    with st.expander(f"结果 {index+1}: {result.get('document_id', '未知文档')} (相关度: {result.get('score', 0):.2f})"):
        st.markdown(f"**来源**: {result.get('source', '未知来源')}")
        st.markdown(f"**内容**:")
        st.markdown(result['content'])

def display_image_result(result: Dict[str, Any], index: int):
    """
    显示图片搜索结果
    
    Args:
        result: 搜索结果
        index: 结果索引
    """
    with st.expander(f"图片 {index+1}: {result.get('document_id', '未知图片')} (相关度: {result.get('score', 0):.2f})"):
        # 获取图片路径
        image_info = result.get('metadata', {}).get('image_info', {})
        s3_path = image_info.get('s3_path', '')
        
        if s3_path:
            # 显示图片
            image_result = get_image_from_s3(s3_path)
            if image_result:
                image, _ = image_result
                st.image(image, caption=f"图片ID: {result['document_id']}", use_container_width=True)
            else:
                st.warning(f"无法加载图片: {s3_path}")
        
        # 显示描述
        st.markdown("**图片描述**:")
        st.markdown(result['content'])

from config import opensearch_config
from get_user_permissions import get_user_permissions

def login_form():
    """显示登录表单"""
    st.title("多模态RAG系统 - 登录")
    
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submit = st.form_submit_button("登录")
        
        if submit:
            if username and password:
                # 认证用户
                auth_result = cognito_auth.authenticate_user(username, password)
                
                if "success" in auth_result:
                    # 认证成功
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.access_token = auth_result.get("access_token")
                    st.session_state.refresh_token = auth_result.get("refresh_token")
                    
                    # 从DynamoDB获取用户权限
                    authorized_docs = get_user_permissions(username)
                    
                    # 如果没有获取到权限，设置为空列表
                    if not authorized_docs:
                        # 对于演示目的，如果是admin用户，给予所有权限
                        if username.lower() == 'admin':
                            st.success("管理员用户，授予所有文档访问权限")
                            authorized_docs = ["*"]
                        else:
                            st.warning(f"警告: 用户 {username} 没有任何文档访问权限")
                            authorized_docs = []
                    
                    # 打印调试信息
                    print(f"用户 {username} 的权限: {authorized_docs}")
                    
                    # 保存用户权限到会话状态
                    st.session_state.authorized_docs = authorized_docs
                    
                    st.success(f"欢迎, {username}!")
                    st.rerun()
                else:
                    # 认证失败
                    st.error(auth_result.get("error", "登录失败，请检查用户名和密码"))
            else:
                st.error("请输入用户名和密码")

def logout():
    """登出用户"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.authorized_docs = []
    st.rerun()

def main():
    """主函数"""
    # 检查用户是否已认证
    if not st.session_state.authenticated:
        login_form()
        return
    
    # 显示用户信息和登出按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("多模态RAG系统")
        
        # 显示用户权限信息
        if st.session_state.authorized_docs == ["*"]:
            st.markdown(f"欢迎, **{st.session_state.username}**! 您有权访问**所有文档**。")
        elif st.session_state.authorized_docs:
            st.markdown(f"欢迎, **{st.session_state.username}**! 您有权访问 **{len(st.session_state.authorized_docs)}** 个文档。")
            with st.expander("查看可访问的文档"):
                for doc in st.session_state.authorized_docs:
                    st.write(f"- {doc}")
        else:
            st.markdown(f"欢迎, **{st.session_state.username}**! 您**没有**权限访问任何文档。")
    
    with col2:
        if st.button("登出"):
            logout()
    
    # 侧边栏配置
    st.sidebar.title("搜索设置")
    
    text_k = st.sidebar.slider("文本结果数量", min_value=1, max_value=10, value=3)
    image_k = st.sidebar.slider("图片结果数量", min_value=1, max_value=5, value=2)
    use_rerank = st.sidebar.checkbox("使用Rerank", value=True)
    use_llm = st.sidebar.checkbox("使用LLM生成回答", value=True)
    
    # 搜索框
    query = st.text_input("输入您的查询", placeholder="例如: VPC对等连接是什么？")
    
    # 搜索按钮
    search_button = st.button("搜索")
    
    # 示例查询
    st.markdown("### 示例查询")
    example_queries = [
        "VPC对等连接是什么？它有什么用途？",
        "AWS架构图中如何表示VPC连接？",
        "如何配置安全组和子网？",
        "应该如何配置Redshift的IAM权限才能正常访问S3"
    ]
    
    # 创建两列布局显示示例查询
    cols = st.columns(2)
    for i, example_query in enumerate(example_queries):
        if cols[i % 2].button(example_query, key=f"example_{i}"):
            query = example_query
            search_button = True
    
    # 执行搜索
    if search_button and query:
        # 检查用户是否有授权文档
        if not st.session_state.authorized_docs:
            st.warning(f"您没有权限访问任何文档。请联系管理员获取权限。")
            return
        
        with st.spinner("正在搜索..."):
            text_results, image_results = combined_search(query, text_k=text_k, image_k=image_k, use_rerank=use_rerank)
        
        # 显示结果
        st.markdown(f"## 搜索结果: '{query}'")
        
        # 创建两列布局
        col1, col2 = st.columns([3, 2])
        
        # 显示文本结果
        with col1:
            st.markdown("### 文本结果")
            if not text_results:
                st.info("没有找到相关文本内容")
            else:
                for i, result in enumerate(text_results):
                    display_text_result(result, i)
        
        # 显示图片结果
        with col2:
            st.markdown("### 图片结果")
            if not image_results:
                st.info("没有找到相关图片")
            else:
                for i, result in enumerate(image_results):
                    display_image_result(result, i)
        
        # 组合回答
        st.markdown("## 组合回答")
        
        if use_llm:
            # 使用LLM生成回答
            with st.spinner("正在生成回答..."):
                answer, references = generate_answer(query, text_results, image_results)
            
            # 显示回答
            st.markdown(answer)
            
            # 显示图片
            if image_results:
                image_info = image_results[0].get('metadata', {}).get('image_info', {})
                s3_path = image_info.get('s3_path', '')
                if s3_path:
                    # 显示图片
                    image_result = get_image_from_s3(s3_path)
                    if image_result:
                        image, _ = image_result
                        # 保存图片到临时文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            image.save(tmp.name)
                            st.image(tmp.name, caption="相关图片", use_container_width=True)
                        # 删除临时文件
                        os.unlink(tmp.name)
            
            # 显示参考资料
            st.markdown("## 参考资料")
            for ref in references:
                st.markdown(f"{ref}")
        else:
            # 简单组合回答
            answer = f"基于您的查询 '{query}'，我找到了以下信息：\n\n"
            
            if text_results:
                answer += "### 文本信息\n\n"
                answer += text_results[0]['content'][:500] + "...\n\n"
            
            st.markdown(answer)

if __name__ == "__main__":
    main()

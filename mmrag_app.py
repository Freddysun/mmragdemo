#!/usr/bin/env python3
"""
多模态RAG系统的Streamlit前端界面
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

from config import bucket_name, region_name, BedrockModels
from opensearch_utils import OpenSearchManager

# 导入combined_search模块中的函数
from combined_search import combined_search, generate_answer

# 初始化客户端
s3_client = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
opensearch_manager = OpenSearchManager()

# 设置页面配置
st.set_page_config(
    page_title="多模态RAG系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("多模态RAG检索系统")
st.markdown("### 同时搜索文本和图片内容")

def get_image_from_s3(s3_path: str) -> Optional[Tuple[Image.Image, bytes]]:
    """
    从S3获取图片
    
    Args:
        s3_path: S3中的图片路径
        
    Returns:
        PIL图片对象和原始图片数据的元组
    """
    try:
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=s3_path
        )
        image_data = response['Body'].read()
        return Image.open(BytesIO(image_data)), image_data
    except Exception as e:
        st.error(f"获取图片时出错: {str(e)}")
        return None

def display_text_result(result: Dict[str, Any], index: int):
    """
    显示文本搜索结果
    
    Args:
        result: 搜索结果
        index: 结果索引
    """
    with st.expander(f"文本结果 {index+1}: {result['source']}", expanded=index==0):
        st.markdown(f"**相关度得分**: {result['score']:.2f}")
        st.markdown(f"**来源**: {result['source']}")
        st.markdown("**内容**:")
        st.markdown(result['content'])

def display_image_result(result: Dict[str, Any], index: int):
    """
    显示图片搜索结果
    
    Args:
        result: 搜索结果
        index: 结果索引
    """
    with st.expander(f"图片结果 {index+1}: {result['document_id']}", expanded=index==0):
        st.markdown(f"**相关度得分**: {result['score']:.2f}")
        st.markdown(f"**来源**: {result['source']}")
        
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

def main():
    """主函数"""
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
            
            if image_results:
                answer += "### 相关图片\n\n"
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
                
                answer += f"图片描述: {image_results[0]['content'][:200]}...\n\n"
            
            st.markdown(answer)

if __name__ == "__main__":
    main()

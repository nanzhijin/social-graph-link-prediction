# Please install OpenAI SDK first: `pip3 install openai`
# test_deepseek_openai.py
import os
from openai import OpenAI

# 设置API Key（选择一种方式）
API_KEY = "sk-8ec70a370eb943d5930e6e9003db29a7"  # ⚠️ 注意安全，不要提交到Git

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    timeout=30.0
)

try:
    print("正在测试DeepSeek API连接...")

    response = client.chat.completions.create(
        model="deepseek-chat",  # 可用模型：deepseek-chat, deepseek-v4-pro
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "请回复'API测试成功'，然后管我叫南志锦"},
        ],
        max_tokens=100,
        temperature=0.7,
        stream=False
    )

    print("✅ 连接成功！")
    print(f"模型回复: {response.choices[0].message.content}")
    print(f"使用模型: {response.model}")
    print(f"消耗token: {response.usage.total_tokens}")

except Exception as e:
    print(f"❌ 连接失败: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
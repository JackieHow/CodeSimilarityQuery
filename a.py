import os
import torch
from unixcoder import UniXcoder
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置模型缓存目录
cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

# 加载 UniXcoder 模型
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)

def embed_code(code):
    # 将代码进行编码
    tokens_ids = model.tokenize([code], max_length=512, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    _, code_embedding = model(source_ids)
    return code_embedding

def calculate_similarity(code1, code2):
    vec1 = embed_code(code1)
    vec2 = embed_code(code2)
    similarity = F.cosine_similarity(vec1, vec2)
    return similarity.item()

# 示例代码
code1 = """

            if (taskList == null || taskList.size() == 0) {
                log.info(\"<------------------ 没有选择该流水线配置分类执行：{} ---------------------->\", testName);
                continue;
            }
"""

code2 = """
            // 获取该配置分类下原先配置的task
"""

# 计算相似度
similarity = calculate_similarity(code1, code2)
print(f"Code similarity: {similarity:.4f}")

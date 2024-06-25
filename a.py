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
    # 归一化嵌入向量
    norm_vec1 = torch.nn.functional.normalize(vec1, p=2, dim=1)
    norm_vec2 = torch.nn.functional.normalize(vec2, p=2, dim=1)
    # 计算余弦相似度
    similarity = torch.einsum("ac,bc->ab", norm_vec1, norm_vec2)
    return similarity.item()

# 示例 Java 代码
code1 = """
            int remain = length;
            int readOnce;
            while (remain > 0 && (readOnce = raf.read(buf, 0, Math.min(remain, buf.length))) != -1) {
                os.write(buf, 0, readOnce);
                remain -= readOnce;
            }
           
"""

code2 = """
            int len = 0;
            int totalLen = 0;
"""

# 计算相似度
similarity = calculate_similarity(code1, code2)
print(f"Code similarity: {similarity:.4f}")

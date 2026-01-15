import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import zlib

def encrypt_code(source_file, output_file, password):
    """
    加密Python代码文件
    
    Args:
        source_file: 要加密的Python源文件路径
        output_file: 加密后输出的文本文件路径
        password: 加密密码（可以是任意字符串或数字）
    """
    
    # 1. 读取原始代码
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    print(f"正在加密文件: {source_file}")
    print(f"原始代码大小: {len(source_code)} 字符")
    
    # 2. 先压缩代码（减少大小）
    compressed = zlib.compress(source_code.encode('utf-8'), level=9)
    print(f"压缩后大小: {len(compressed)} 字节")
    
    # 3. 使用密码生成密钥
    # 使用SHA256生成32字节密钥（AES-256需要32字节）
    key = hashlib.sha256(str(password).encode()).digest()
    
    # 4. 生成随机IV（初始化向量）
    iv = get_random_bytes(16)
    
    # 5. 创建AES加密器（CBC模式）
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # 6. 加密数据（需要填充）
    encrypted_data = cipher.encrypt(pad(compressed, AES.block_size))
    
    # 7. 组合IV和加密数据，并进行Base64编码
    # 格式: IV(16字节) + 加密数据
    combined = iv + encrypted_data
    encoded_data = base64.b64encode(combined).decode('utf-8')
    
    # 8. 保存为文本文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(encoded_data)
    
    print(f"加密完成！输出文件: {output_file}")
    print(f"加密后文本大小: {len(encoded_data)} 字符")
    print("\n" + "="*50)
    print("加密文本预览（前200字符）:")
    print(encoded_data[:200] + "...")
    print("="*50)
    print("\n提示：你可以复制上面的加密文本，在其他电脑上使用解密脚本还原。")

if __name__ == "__main__":
    # 安装依赖（如果尚未安装）:
    # pip install pycryptodome
    
    # 使用示例
    source_file = "test.py"  # 替换为你的Python文件
    output_file = "encrypted_code.txt"  # 加密后的文本文件
    password = "102938"  # 你的密钥
    
    encrypt_code(source_file, output_file, password)

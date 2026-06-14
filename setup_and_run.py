"""在 AutoDL 5090 上配置环境并运行 gnn_baseline_B"""
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 34885
USER = "root"
PASSWORD = "xuxAxBMg97ao"
WORKDIR = "/root/autodl-tmp"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASSWORD, timeout=30)

def run(cmd, desc=""):
    if desc:
        print(f"\n=== {desc} ===")
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=300)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out:
        print(out)
    if err:
        print(f"ERR: {err}")
    return out, err

# 1. 检查 CUDA 和 PyTorch
run("python -c 'import torch; print(f\"PyTorch {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0)}\")'",
    "检查环境")

# 2. 安装 torch_geometric
run("pip install torch_geometric --no-deps -q 2>&1 | tail -5",
    "安装 torch_geometric")

# 3. 安装 torch_scatter 等依赖
run("pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html -q 2>&1 | tail -10",
    "安装 pyg 扩展")

# 4. 验证 torch_geometric
run("python -c 'import torch_geometric; print(f\"PyG {torch_geometric.__version__} OK\")'",
    "验证 PyG")

# 5. 运行 gnn_baseline_B
print("\n" + "="*60)
print("🔥 启动 GNN-B 训练 (RTX 5090)")
print("="*60)
run(f"cd {WORKDIR}/3_算法建模 && python gnn_baseline_B.py 2>&1",
    "运行 gnn_baseline_B")

ssh.close()
print("\n完成!")

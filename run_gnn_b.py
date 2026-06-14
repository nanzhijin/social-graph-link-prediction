"""在 AutoDL 5090 上跑 GNN-B"""
import paramiko, time

HOST = "connect.bjb2.seetacloud.com"
PORT = 34885
USER = "root"
PASSWORD = "xuxAxBMg97ao"
WD = "/root/autodl-tmp"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASSWORD, timeout=30)
print("SSH connected")

def run(cmd, timeout=300):
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out.strip():
        print(out)
    if err.strip():
        print(f"[E] {err}")
    return out, err

# 1. 探测 Python - Ubuntu 22.04 用 python3
print("\n=== Step 1: 探测 Python ===")
out, _ = run("command -v python3; command -v python")

# 2. 验证 PyTorch
print("\n=== Step 2: PyTorch + CUDA ===")
run('python3 -c "import torch; print(torch.__version__); print(\'CUDA:\', torch.cuda.is_available()); print(\'GPU:\', torch.cuda.get_device_name(0))"')

# 3. 装 PyG
print("\n=== Step 3: pip install torch_geometric ===")
run("python3 -m pip install torch_geometric -q 2>&1 | tail -5")

# 4. 验证
print("\n=== Step 4: 验证 PyG ===")
run('python3 -c "import torch_geometric; print(\'PyG:\', torch_geometric.__version__)"')

# 5. 跑训练 (用 shell 获得实时输出)
print("\n=== Step 5: 启动训练 ===")
channel = ssh.invoke_shell()
channel.settimeout(10)
channel.send(f"cd {WD}/3_算法建模 && python3 gnn_baseline_B.py\n")
time.sleep(3)

deadline = time.time() + 600
try:
    while time.time() < deadline:
        if channel.recv_ready():
            data = channel.recv(65536).decode('utf-8', errors='replace')
            print(data, end='', flush=True)
        else:
            time.sleep(2)
except KeyboardInterrupt:
    channel.send('\x03')
finally:
    channel.close()

ssh.close()
print("\n=== Done ===")

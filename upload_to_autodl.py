"""上传 gnn_b_cloud.tar 到 AutoDL 5090 实例"""
import paramiko
from scp import SCPClient
import os

HOST = "connect.bjb2.seetacloud.com"
PORT = 34885
USER = "root"
PASSWORD = "xuxAxBMg97ao"
LOCAL_FILE = r"D:\GNN\gnn_b_cloud.tar"
REMOTE_DIR = "/root/autodl-tmp/"

print(f"连接 {HOST}:{PORT}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, PORT, USER, PASSWORD, timeout=30)
print("SSH 已连接")

# 上传
print(f"上传 {os.path.getsize(LOCAL_FILE)/1024**3:.1f}GB...")
with SCPClient(ssh.get_transport(), socket_timeout=300) as scp:
    scp.put(LOCAL_FILE, REMOTE_DIR, recursive=False)
print("上传完成!")

# 验证 + 解压
stdin, stdout, stderr = ssh.exec_command(
    f"cd {REMOTE_DIR} && ls -lh gnn_b_cloud.tar && tar -xf gnn_b_cloud.tar"
)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print(f"STDERR: {err}")

# 检查目录结构
stdin, stdout, stderr = ssh.exec_command(
    f"ls -d {REMOTE_DIR}3_算法建模 {REMOTE_DIR}modules {REMOTE_DIR}modules/models {REMOTE_DIR}processed"
)
print(f"目录验证:\n{stdout.read().decode()}")

print("全部就绪!")
ssh.close()

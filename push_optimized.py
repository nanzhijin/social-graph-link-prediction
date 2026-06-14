"""推送修复后的 gnn_model_B.py"""
import paramiko
from scp import SCPClient

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.bjb2.seetacloud.com", 34885, "root", "xuxAxBMg97ao", timeout=30)

with SCPClient(ssh.get_transport(), socket_timeout=30) as scp:
    scp.put(
        r"D:\GNN\modules\models\gnn_model_B.py",
        "/root/autodl-tmp/modules/models/gnn_model_B.py"
    )

# 验证改动
_, out, _ = ssh.exec_command(
    "grep -n 'emb_all_b\|emb_all_a\|cand_emb_b\|inv_emb_b\|cand_emb_a\|inv_emb_a' /root/autodl-tmp/modules/models/gnn_model_B.py"
)
print(out.read().decode())

ssh.close()
print("Done!")

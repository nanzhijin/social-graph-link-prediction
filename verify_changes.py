"""验证优化后的代码"""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.bjb2.seetacloud.com", 34885, "root", "xuxAxBMg97ao", timeout=30)

# 检查3处改动
_, stdout, _ = ssh.exec_command(
    "grep -n 'recent_pair_set\|voter_cate_score_dict\|voter_top3_set' /root/autodl-tmp/modules/models/gnn_model_B.py"
)
print("=== 改动的3处 dict 变量 ===")
print(stdout.read().decode())

# 检查是否还有老代码
_, stdout, _ = ssh.exec_command(
    "grep -n 'len(pr) > 0\|len(match_row) > 0\|len(top3_row) > 0' /root/autodl-tmp/modules/models/gnn_model_B.py || echo '(老代码已清除)'"
)
print("=== 老 pandas filter 是否还存在 ===")
print(stdout.read().decode())

ssh.close()

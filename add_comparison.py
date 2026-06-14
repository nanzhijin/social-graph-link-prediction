"""在服务器端添加 GNN-A 直接 MRR 对比代码并运行"""
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("connect.bjb2.seetacloud.com", 34885, "root", "xuxAxBMg97ao", timeout=30)

# 在 gnn_baseline_B.py 末尾追加对比代码
add_code = '''
# ============================================================
# 6. 验证: GNN-A 用原始 evaluate_mrr 直接评估 (不经过 GNNModelB)
# ============================================================
print("\\n" + "=" * 60)
print("6. 验证: GNN-A 原始 evaluate_mrr (对照基准)")
print("=" * 60)
result_a_direct = gnn_a.evaluate_mrr(valid, n_queries=500)

print("\\n" + "=" * 60)
print("GNN-B 假设检验完成!")
print(f"  AUC:  A={auc_a_valid:.4f}  B={auc_b_valid:.4f}  d={auc_b_valid-auc_a_valid:+.4f}")
print(f"  MRR(GNN-B eval):  A={result_b.get('mrr_global_a',0):.5f}  B={result_b['mrr_global']:.5f}")
print(f"         A_friend={result_b.get('mrr_friend_a',0):.5f} B_friend={result_b['mrr_friend']:.5f}")
print(f"         A_stranger={result_b.get('mrr_stranger_a',0):.5f} B_stranger={result_b['mrr_stranger']:.5f}")
print(f"  MRR(GNN-A direct): A={result_a_direct['mrr@5']:.5f}  HITS={result_a_direct['hits@5']:.5f}")
print("=" * 60)
'''

# 先把原始文件末尾那些老 print 去掉，然后追加新的
cmd = f"""cd /root/autodl-tmp/3_算法建模 && python3 -c "
lines = open('gnn_baseline_B.py', 'r', encoding='utf-8').readlines()
# 找到最后的 print('\\\\n' + '=' * 60) 之前的行
cut = None
for i in range(len(lines)-1, -1, -1):
    if 'GNN-B 假设检验完成' in lines[i] and cut is None:
        cut = i
        break
# 找到上一行 print
if cut:
    for j in range(cut-1, -1, -1):
        if 'print' in lines[j] and '=' * 60 in lines[j]:
            cut = j
            break

# 去掉尾部多余代码，保留到 '5. 保存结果' 之前
for i in range(len(lines)-1, -1, -1):
    if '5. 保存结果' in lines[i]:
        cut = i
        break

if cut:
    lines = lines[:cut]

# Append comparison code
add_code = '''{add_code}'''
lines.append(add_code)
open('gnn_baseline_B.py', 'w', encoding='utf-8').writelines(lines)
print('File updated OK')
with open('gnn_baseline_B.py', 'r') as f:
    print(f.readlines()[-15:])
"
"""
ssh.exec_command(cmd, timeout=30)
out = ssh.exec_command("tail -20 /root/autodl-tmp/3_算法建模/gnn_baseline_B.py")[1].read().decode()
print(out)

ssh.close()

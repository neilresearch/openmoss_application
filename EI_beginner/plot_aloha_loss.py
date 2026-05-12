import re
import matplotlib.pyplot as plt

log_path = "aloha_full_training.log"
steps = []
losses = []

# 定义匹配模式：提取 step 和 loss
# 样例：step:100K ... loss:0.018
pattern = re.compile(r"step:(\d+)(K?)\s+.*loss:([\d.]+)")

with open(log_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            # 处理 step (如果是 10K 转换为 10000)
            step_val = int(match.group(1))
            if match.group(2) == 'K':
                step_val *= 1000
            
            loss_val = float(match.group(3))
            steps.append(step_val)
            losses.append(loss_val)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label='Training Loss')
plt.title('Aloha Training Loss Curve')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 保存图片
plt.savefig('aloha_loss_curve.png')
print(f"解析完成，已处理 {len(steps)} 个数据点。曲线图已保存至 aloha_loss_curve.png")
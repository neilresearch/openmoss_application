import re
import matplotlib.pyplot as plt

experiments = {
    "exp1_baseline.log": "Baseline (obs=2, horizon=16)",
    "exp2_obs4.log": "Long Obs (obs=4)",
    "exp3_horizon4.log": "Short Horizon (horizon=4)",
    "exp4_horizon24.log": "Long Horizon (horizon=24)",
    "exp5_act_baseline.log": "ACT Model (Transformer)"
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def smooth_curve(scalars, weight=0.85):
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

data_store = {name: {'step_loss': [], 'loss': [], 'step_eval': [], 'success': [], 'reward': []} for name in experiments.values()}

regex_loss = re.compile(r"step:(\d+)(K?)\s+.*loss:([\d.]+)")
regex_eval = re.compile(r"step:\s*(\d+)(K?)[^\n]*?eval/success_rate[^\d]*([\d.]+)")

for log_file, label in experiments.items():
    try:
        with open(log_file, "r") as f:
            for line in f:
                m_step_train = re.search(r"step:(\d+)(K?)", line)
                if m_step_train:
                    current_step = int(m_step_train.group(1)) * (1000 if m_step_train.group(2) == 'K' else 1)
                
                m_step_video = re.search(r"videos_step_(\d+)", line)
                if m_step_video:
                    current_step = int(m_step_video.group(1))
                    
                m_loss = re.search(r"loss:([\d.]+)", line)
                if m_loss and m_step_train:
                    loss_val = float(m_loss.group(1))
                    data_store[label]['step_loss'].append(current_step)
                    data_store[label]['loss'].append(loss_val)
                
                m_eval = re.search(r"['\"]pc_success['\"]\s*:\s*([\d.]+)", line)
                if m_eval:
                    success_val = float(m_eval.group(1))
                    
                    if not data_store[label]['step_eval'] or data_store[label]['step_eval'][-1] != current_step:
                        data_store[label]['step_eval'].append(current_step)
                        data_store[label]['success'].append(success_val)
    except FileNotFoundError:
        print(f"警告：未找到文件 {log_file}")

plt.style.use('seaborn-v0_8-whitegrid') 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# 子图 1: 平滑后的 Loss 曲线
for idx, (label, data) in enumerate(data_store.items()):
    if data['step_loss']:
        smoothed_loss = smooth_curve(data['loss'], weight=0.9) 
        ax1.plot(data['step_loss'], smoothed_loss, label=label, color=colors[idx], linewidth=2, alpha=0.9)

ax1.set_title('Training Loss (Smoothed)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Loss (MSE / L1)', fontsize=12)
ax1.set_ylim(0, 0.15) 
ax1.legend(fontsize=10)

# 子图 2: Success Rate 曲线
for idx, (label, data) in enumerate(data_store.items()):
    if data['step_eval']:
        ax2.plot(data['step_eval'], data['success'], label=label, color=colors[idx], marker='o', linewidth=2, markersize=6)

ax2.set_title('Evaluation Success Rate', fontsize=14, fontweight='bold')
ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Success Rate (0.0 to 1.0)', fontsize=12)
ax2.set_ylim(-5, 105)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('pusht_ablation_results.png')
print("图表生成完毕！已保存为 openmoss_ablation_results.png (300 DPI 超清版)")
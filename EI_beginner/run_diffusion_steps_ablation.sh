#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

echo "==========================================="
echo "开始实验 1：Baseline (标准配置: obs=2, horizon=16)"
echo "==========================================="
xvfb-run -a lerobot-train \
  --policy.type=diffusion \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --steps=40000 \
  --save_freq=40000 \
  --eval_freq=10000 \
  --save_checkpoint=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --dataset.video_backend=pyav \
  --output_dir=~/outputs/exp1_baseline \
  2>&1 | tee ~/outputs/exp1_baseline.log

echo "==========================================="
echo "开始实验 2：增加观测 (增加历史视野: obs=4)"
echo "==========================================="
# n_obs_steps=4 意味着让模型记住更长的时间序列，更好感知动量
xvfb-run -a lerobot-train \
  --policy.type=diffusion \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --steps=40000 \
  --save_freq=40000 \
  --eval_freq=10000 \
  --save_checkpoint=true \
  --policy.n_obs_steps=4 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --dataset.video_backend=pyav \
  --output_dir=~/outputs/exp2_obs4 \
  2>&1 | tee ~/outputs/exp2_obs4.log

echo "==========================================="
echo "开始实验 3：短视动作 (消融预测步长: horizon=4)"
echo "==========================================="
# horizon=4 & n_action_steps=4 削弱动作分块，测试轨迹的连贯性
xvfb-run -a lerobot-train \
  --policy.type=diffusion \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --steps=40000 \
  --save_freq=40000 \
  --eval_freq=10000 \
  --save_checkpoint=true \
  --policy.horizon=4 \
  --policy.n_action_steps=4 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --dataset.video_backend=pyav \
  --output_dir=~/outputs/exp3_horizon4 \
  2>&1 | tee ~/outputs/exp3_horizon4.log

echo "==========================================="
echo "开始实验 4：增加预测步长 (增加预测步长: horizon=24)"
echo "==========================================="
# horizon=24 增加模型开环规划的距离，测试超长预测是否会发散
xvfb-run -a lerobot-train \
  --policy.type=diffusion \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --steps=40000 \
  --save_freq=40000 \
  --eval_freq=10000 \
  --save_checkpoint=true \
  --policy.horizon=24 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --dataset.video_backend=pyav \
  --output_dir=~/outputs/exp4_horizon24 \
  2>&1 | tee ~/outputs/exp4_horizon24.log
  
echo "==========================================="
echo "开始实验 5：对比 Diffusion 和 ACT"
echo "==========================================="
# 引入强大的 ACT 作为 Baseline 对手
xvfb-run -a lerobot-train \
  --policy.type=act \
  --env.type=pusht \
  --dataset.repo_id=lerobot/pusht \
  --steps=40000 \
  --save_freq=40000 \
  --eval_freq=10000 \
  --save_checkpoint=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --eval.use_async_envs=false \
  --eval.batch_size=5 \
  --eval.n_episodes=10 \
  --dataset.video_backend=pyav \
  --output_dir=~/outputs/exp5_act_baseline \
  2>&1 | tee ~/outputs/exp5_act_baseline.log

echo "所有消融实验执行完毕！"
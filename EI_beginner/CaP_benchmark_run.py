import pybullet as p
import pybullet_data
import time
import numpy as np
import random
from datetime import datetime
import json
import re
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
RUN_MODE = "benchmark"       
GUI_MODE = False             # 切换评测目标: "deepseek" | "qwen-base-zero" | "qwen-base-few" | "qwen-ft"
EVAL_MODEL = "qwen-base-few"       
NUM_TRIALS = 50

LOCAL_BASE_MODEL_PATH = "/root/autodl-tmp/model_cache/modelscope/hub/models/qwen/Qwen2.5-7B" # 替换为真实基座路径
LOCAL_FT_MODEL_PATH = "/root/autodl-tmp/my_robot_model_merged"    # 替换为真实微调导出路径

# ================= 全局变量与模型初始化 =================
object_lookup = {}
EE_INDEX = 11  
global_grasp_constraint = None

client = OpenAI(api_key="sk-8e0f9b9028cb4646a2116239e7531aa9", base_url="https://api.deepseek.com")
local_model = None
local_tokenizer = None

def init_local_model(model_type):
    global local_model, local_tokenizer
    print(f"\n[系统] 正在加载本地模型: {model_type}...")
    # 只要带有 'base' 的，都去加载原始模型；否则加载微调模型
    model_path = LOCAL_FT_MODEL_PATH if model_type == "qwen-ft" else LOCAL_BASE_MODEL_PATH
    
    local_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    local_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    print("[系统] 模型加载完成！")

# ================= 物理引擎与原子控制 (保持原有逻辑) =================
def get_downward_orn(): return p.getQuaternionFromEuler([0, np.pi, 0])

def step_env(duration_sec):
    steps = int(duration_sec * 240)
    for _ in range(steps):
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

def get_movable_joints(robot_id):
    movable_joints, lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], [], []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] != p.JOINT_FIXED:
            movable_joints.append(i)
            lower_limits.append(info[8])
            upper_limits.append(info[9])
            joint_ranges.append(info[9] - info[8])
            rest_poses.append(p.getJointState(robot_id, i)[0])
    return movable_joints, lower_limits, upper_limits, joint_ranges, rest_poses

def move_to(robot_id, ee_index, target_pos, target_orn=None, steps=100, use_rest_poses=True):
    joints, ll, ul, jr, _ = get_movable_joints(robot_id)
    current_ee_state = p.getLinkState(robot_id, ee_index)
    start_pos = np.array(current_ee_state[0])
    if target_orn is None: target_orn = current_ee_state[1]
    custom_rp = [0, -0.6, 0, -2.0, 0, 1.5, 0.8]
    gain = 0.05 if use_rest_poses else 0.2
    for i in range(steps):
        t = (i + 1) / steps
        intermediate_pos = start_pos + t * (np.array(target_pos[:3]) - start_pos)
        if use_rest_poses:
            joint_poses = p.calculateInverseKinematics(
                robot_id, ee_index, intermediate_pos, targetOrientation=target_orn,
                lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=custom_rp, residualThreshold=1e-5, maxNumIterations=100)
        else:
            joint_poses = p.calculateInverseKinematics(
                robot_id, ee_index, intermediate_pos, targetOrientation=target_orn,
                residualThreshold=1e-5, maxNumIterations=100)
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id, jointIndices=joints, controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses, forces=[500.0] * len(joints),
            targetVelocities=[0.5] * len(joints), positionGains=[gain] * len(joints))
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)
    for _ in range(40):
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id, jointIndices=joints, controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses, forces=[500.0] * len(joints), positionGains=[0.2] * len(joints))
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

def gripper_control(robot_id, target_pos, force=20, steps=100):
    finger_indices = [9, 10]
    for _ in range(steps):
        p.setJointMotorControlArray(robot_id, finger_indices, p.POSITION_CONTROL, targetPositions=[target_pos, target_pos], forces=[force, force])
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

def grasp(robot_id, force=20):
    global global_grasp_constraint
    gripper_control(robot_id, target_pos=0.0, force=force)
    ee_pos = p.getLinkState(robot_id, EE_INDEX)[0]
    for bid, label in object_lookup.items():
        if "block" in label:
            obj_pos, _ = p.getBasePositionAndOrientation(bid)
            dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
            if dist < 0.06:
                global_grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=robot_id, parentLinkIndex=EE_INDEX, childBodyUniqueId=bid, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0.03], childFramePosition=[0, 0, 0])
                break

def release(robot_id):
    global global_grasp_constraint
    if global_grasp_constraint is not None:
        p.removeConstraint(global_grasp_constraint)
        global_grasp_constraint = None
    gripper_control(robot_id, target_pos=0.04, force=100)

def is_grabbed(robot_id): return (p.getJointState(robot_id, 9)[0] + p.getJointState(robot_id, 10)[0]) > 0.01
def register_object(body_id, label): object_lookup[body_id] = label
def get_object_coordinates(): return {label: list(p.getBasePositionAndOrientation(bid)[0]) for bid, label in object_lookup.items() if bid > 1}
def get_object_pose(label):
    for bid, name in object_lookup.items():
        if name == label: return list(p.getBasePositionAndOrientation(bid)[0]), list(p.getBasePositionAndOrientation(bid)[1])
    return None, None

def simple_pick(robot_id, obj_pos, grip_orn=None):
    if grip_orn is None: grip_orn = get_downward_orn()
    hover_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15] 
    move_to(robot_id, EE_INDEX, hover_pos, grip_orn)
    release(robot_id)
    step_env(0.5) 
    current_coords = get_object_coordinates()
    actual_pos = obj_pos 
    for label, pos in current_coords.items():
        if np.linalg.norm(np.array(pos[:2]) - np.array(obj_pos[:2])) < 0.05:
            actual_pos = pos; break
    actual_hover_pos = [actual_pos[0], actual_pos[1], actual_pos[2] + 0.15]
    move_to(robot_id, EE_INDEX, actual_hover_pos, grip_orn, use_rest_poses=False)
    grasp_pos = [actual_pos[0], actual_pos[1], actual_pos[2] + 0.025]
    move_to(robot_id, EE_INDEX, grasp_pos, grip_orn, use_rest_poses=False)
    grasp(robot_id)
    step_env(0.5) 
    if is_grabbed(robot_id): move_to(robot_id, EE_INDEX, [actual_pos[0], actual_pos[1], actual_pos[2] + 0.2], grip_orn, use_rest_poses=False)
    else: release(robot_id); move_to(robot_id, EE_INDEX, actual_hover_pos, grip_orn, use_rest_poses=False)

def stack(block1, block2):
    pos1, orn1 = get_object_pose(block1)
    pos2, _ = get_object_pose(block2)
    if not pos1 or not pos2: return
    euler1 = p.getEulerFromQuaternion(orn1)
    dynamic_orn = p.getQuaternionFromEuler([0, np.pi, euler1[2]])
    simple_pick(panda_id, pos1, grip_orn=dynamic_orn)
    hover_pos = np.array(pos2) + np.array([0, 0, 0.2])
    move_to(panda_id, EE_INDEX, hover_pos, dynamic_orn)
    stack_pos = np.array(pos2) + np.array([0, 0, 0.06])
    move_to(panda_id, EE_INDEX, stack_pos, dynamic_orn, use_rest_poses=False)
    release(panda_id)
    move_to(panda_id, EE_INDEX, hover_pos, get_downward_orn(), use_rest_poses=False)

def place_in_bowl(block, bowl):
    pos1, orn1 = get_object_pose(block)
    bowl_pos, _ = get_object_pose(bowl)
    if not pos1 or not bowl_pos: return
    euler1 = p.getEulerFromQuaternion(orn1)
    dynamic_orn = p.getQuaternionFromEuler([0, np.pi, euler1[2]])
    simple_pick(panda_id, pos1, grip_orn=dynamic_orn)
    drop_pos = np.array(bowl_pos) + np.array([0, 0, 0.15])
    move_to(panda_id, EE_INDEX, drop_pos, dynamic_orn)
    release(panda_id)

COLORS = {'red': [1, 0, 0, 1], 'green': [0, 1, 0, 1], 'blue': [0, 0, 1, 1]}
def create_flat_plate(pos, label, rgba):
    radius, height = 0.15, 0.005
    shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
    plate_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape_id, baseVisualShapeIndex=visual_id, basePosition=[pos[0], pos[1], height/2])
    register_object(plate_id, label)
    return plate_id

def reset_random_scene():
    for body_id in list(object_lookup.keys()):
        if body_id not in [0, panda_id]: p.removeBody(body_id)
    object_lookup.clear()
    create_flat_plate([random.uniform(0.35, 0.55), random.uniform(0.2, 0.4), 0], "bowl1", rgba=[0.1, 0.1, 0.3, 1])
    create_flat_plate([random.uniform(0.35, 0.55), random.uniform(-0.4, -0.2), 0], "bowl2", rgba=[0.3, 0.1, 0.1, 1])
    spawn_zones = {'red': {'x': [0.35, 0.45], 'y': [0.15, 0.25]}, 'green': {'x': [0.35, 0.45], 'y': [-0.05, 0.05]}, 'blue': {'x': [0.35, 0.45], 'y': [-0.25, -0.15]}}
    for color, rgba in COLORS.items():
        zone = spawn_zones[color]
        pos = [random.uniform(zone['x'][0], zone['x'][1]), random.uniform(zone['y'][0], zone['y'][1]), 0.1]
        cube_id = p.loadURDF("cube.urdf", pos, globalScaling=0.05)
        p.changeVisualShape(cube_id, -1, rgbaColor=rgba)
        p.changeDynamics(cube_id, -1, mass=0.1, lateralFriction=100.0, spinningFriction=10.0, rollingFriction=10.0, contactProcessingThreshold=0.005)
        register_object(cube_id, f"{color}_block")
    step_env(1.5)
    return get_object_coordinates()

def evaluate_task(instruction):
    coords = get_object_coordinates()
    try:
        blocks = re.findall(r'(red_block|green_block|blue_block)', instruction.lower())
        bowls = re.findall(r'(bowl1|bowl2)', instruction.lower())
        if bowls and blocks:
            target_block, target_bowl = blocks[0], bowls[0]
            dist = np.linalg.norm(np.array(coords[target_block][:2]) - np.array(coords[target_bowl][:2]))
            return dist < 0.15 and coords[target_block][2] < 0.1
        elif len(blocks) >= 2 and not bowls:
            top_block, base_block = blocks[0], blocks[1]
            top_pos, base_pos = coords[top_block], coords[base_block]
            xy_dist = np.linalg.norm(np.array(top_pos[:2]) - np.array(base_pos[:2]))
            z_diff = top_pos[2] - base_pos[2]
            return xy_dist < 0.05 and 0.04 < z_diff < 0.08
    except Exception: return False
    return False

# ================= 全局 Few-Shot 示例库 (CaP 范式) =================
# ================= 全局 Few-Shot 示例库 (严格遵循 CoT 与 Markdown 规范) =================
FEW_SHOT_EXAMPLES = """
# Example 1: 把红方块向右平移 0.1 米 (基础抓取)
Thinking: The user wants to move the red block. I need to get its coordinates, add an offset to the Y-axis, and execute a simple pick and move operation.
```python
block_pos = get_object_coordinates()['red_block']
target_pos = np.array(block_pos) + np.array([0, -0.1, 0])
simple_pick(panda_id, block_pos, grip_orn=get_downward_orn())
move_to(panda_id, EE_INDEX, target_pos, get_downward_orn())
release(panda_id)
```

# Example 2: 如果绿方块在一号碗里，把它拿到安全的空地上 (高级姿态抓取)
Thinking: I need to check the distance between the green block and bowl1. If it's close, I will calculate the dynamic grasp orientation and move it to a safe coordinate.
```python
block_pos = get_object_coordinates()['green_block']
bowl_pos = get_object_coordinates()['bowl1']
if np.linalg.norm(np.array(block_pos[:2]) - np.array(bowl_pos[:2])) < 0.15:
    _, orn = get_object_pose('green_block')
    yaw = p.getEulerFromQuaternion(orn)[2]
    dyn_orn = p.getQuaternionFromEuler([0, np.pi, yaw])
    simple_pick(panda_id, block_pos, grip_orn=dyn_orn)
    move_to(panda_id, EE_INDEX, [0.4, 0.0, 0.1], dyn_orn)
    release(panda_id)
```

# Example 3: 把 red_block 堆叠到 blue_block 上
Thinking: The user wants to stack two blocks. I can directly use the provided high-level 'stack' function.
```python
stack('red_block', 'blue_block')
```

# Example 4: 将 green_block 放入 bowl2 中
Thinking: The task is to place a specific block into a specific bowl. I will use the 'place_in_bowl' function.
```python
place_in_bowl('green_block', 'bowl2')
```
"""

# ================= 推理执行与耗时统计 =================
def llm_reasoning_and_execution(instruction, env_state, model_type):
    base_sys_p = """You are an Embodied AI planner controlling a Panda arm.
    Available Python functions:
    - get_object_coordinates() -> dict
    - stack(block1, block2)
    - place_in_bowl(block, bowl)
    - simple_pick(robot_id, pos): Moves to pos, descends, and grasps.
    - move_to(robot_id, ee_index, pos, target_orn): Moves the grasped object to a new pos.
    - release(robot_id): Opens the gripper.
    - is_grabbed(robot_id) -> bool: Returns True if an object is secured.
    You must reply with TWO parts:
    1. "Thinking:" A short chain of thought (CoT) analyzing spatial locations.
    2. The exact Python code wrapped in ```python ... ```."""
    
    # 严格分配 Prompt 策略：只有 deepseek 和 qwen-base-few 获得示例
    if model_type in ["deepseek", "qwen-base-few"]:
        sys_p = base_sys_p + "\n\nHere are some code examples:\n" + FEW_SHOT_EXAMPLES
    else:
        # qwen-base-zero 和 qwen-ft 只能硬着头皮做 Zero-shot
        sys_p = base_sys_p

    user_p = f"Current State: {env_state}\nTask: {instruction}\nWrite the plan and code."
    
    start_time = time.time()
    
    if model_type == "deepseek":
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
            temperature=0.1
        )
        raw_output = response.choices[0].message.content
        
    elif "qwen" in model_type: # 匹配 qwen-base-zero, qwen-base-few, qwen-ft
        messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}]
        text = local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = local_tokenizer([text], return_tensors="pt").to(local_model.device)
        
        with torch.no_grad():
            generated_ids = local_model.generate(**model_inputs, max_new_tokens=512, temperature=0.1)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            raw_output = local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    inference_time = time.time() - start_time
    
    code_match = re.search(r'```python\s+(.*?)\s+```', raw_output, re.DOTALL)
    if not code_match: 
        print(f"❌ 正则提取失败。模型原始输出如下：\n{raw_output[:200]}...") # 打印开头部分方便查错
        return raw_output, None, inference_time
        
    code_to_exec = code_match.group(1).strip()
    try:
        exec(code_to_exec, globals(), globals())
        step_env(2.0) 
        return raw_output, code_to_exec, inference_time
    except Exception as e:
        print(f"⚠️ 代码执行报错: {e}")
        return raw_output, None, inference_time

# ================= 基准测试核心逻辑 =================
def run_benchmark(model_type, num_trials):
    print(f"\n{'='*50}\n🚀 开始跑分测试: {model_type}\n测试轮数: {num_trials}\n{'='*50}")
    
    tasks_pool = [
        ("pick_place", "Put the red_block in bowl1."),
        ("pick_place", "Place the green_block into bowl2."),
        ("pick_place", "Without hitting anything, place the blue_block in bowl2."),
        ("stack", "Stack the green_block on the red_block."),
        ("stack", "Carefully stack the blue_block on the red_block."),
        ("stack", "Locate the blue_block and stack it on the red_block.")
    ]
    
    # 增加详细日志记录结构
    report_data = {
        "model": model_type,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_trials": num_trials,
        "summary": {
            "pick_place": {"success": 0, "total": 0, "avg_time_sec": 0.0},
            "stack": {"success": 0, "total": 0, "avg_time_sec": 0.0}
        },
        "trial_details": [] # 记录每一步的具体表现
    }
    
    results = {"pick_place": {"success": 0, "total": 0, "times": []}, 
               "stack": {"success": 0, "total": 0, "times": []}}
    
    for i in range(num_trials):
        task_type, instruction = random.choice(tasks_pool)
        print(f"\n[Trial {i+1}/{num_trials}] Task: {instruction}")
        
        env_state = reset_random_scene()
        raw_code, executed_code, inf_time = llm_reasoning_and_execution(instruction, env_state, model_type)
        
        is_success = False
        if executed_code and evaluate_task(instruction):
            is_success = True
            
        results[task_type]["total"] += 1
        results[task_type]["times"].append(inf_time)
        if is_success: results[task_type]["success"] += 1
        
        # 记录单条测试详情
        report_data["trial_details"].append({
            "trial_id": i + 1,
            "task_type": task_type,
            "instruction": instruction,
            "success": is_success,
            "inference_time": round(inf_time, 3)
        })
        
        print(f"结果: {'✅ 成功' if is_success else '❌ 失败'} | 推理耗时: {inf_time:.2f}s")
    
    print(f"\n\n📊 【{model_type} 最终评测报告】 📊")
    for t_type, stats in results.items():
        if stats["total"] > 0:
            success_rate = (stats["success"] / stats["total"]) * 100
            avg_time = np.mean(stats["times"])
            
            # 更新到保存字典中
            report_data["summary"][t_type]["success"] = stats["success"]
            report_data["summary"][t_type]["total"] = stats["total"]
            report_data["summary"][t_type]["avg_time_sec"] = round(avg_time, 3)
            report_data["summary"][t_type]["success_rate_percent"] = round(success_rate, 2)
            
            print(f"- [{t_type}] 成功率: {success_rate:.1f}% ({stats['success']}/{stats['total']}) | 平均耗时: {avg_time:.2f} 秒")

    # 结果持久化保存到本地 JSON 文件
    filename = f"benchmark_report_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=4)
        print(f"\n💾 评测结果已成功保存至本地: {filename}")
    except Exception as e:
        print(f"\n⚠️ 保存结果文件失败: {e}")

# ================= 主入口 =================
if __name__ == "__main__":
    p.connect(p.GUI if GUI_MODE else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    movable_joints, _, _, _, _ = get_movable_joints(panda_id)
    custom_rp = [0, -0.6, 0, -2.0, 0, 1.5, 0.8] 
    for i in range(len(custom_rp)): p.resetJointState(panda_id, movable_joints[i], custom_rp[i])
    p.changeDynamics(panda_id, 9, lateralFriction=100.0, spinningFriction=10.0)
    p.changeDynamics(panda_id, 10, lateralFriction=100.0, spinningFriction=10.0)

    if RUN_MODE == "benchmark":
        # 只要名字里带 qwen 的，都给我去加载本地模型！
        if "qwen" in EVAL_MODEL:
            init_local_model(EVAL_MODEL)
        run_benchmark(EVAL_MODEL, num_trials=NUM_TRIALS)

    p.disconnect()
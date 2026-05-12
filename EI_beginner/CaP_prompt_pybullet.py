
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import json
import re
from openai import OpenAI


RUN_MODE = "collection" # "inference" 或 "collection" 或 "test_prompt"

GUI_MODE = True
object_lookup = {}
EE_INDEX = 11  
global_grasp_constraint = None

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

def get_downward_orn():
    return p.getQuaternionFromEuler([0, np.pi, 0])

def step_env(duration_sec):
    steps = int(duration_sec * 240)
    for _ in range(steps):
        p.stepSimulation()
        if GUI_MODE:
            time.sleep(1./240.)

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
                lowerLimits=ll, upperLimits=ul, jointRanges=jr, 
                restPoses=custom_rp, residualThreshold=1e-5, maxNumIterations=100
            )
        else:
            joint_poses = p.calculateInverseKinematics(
                robot_id, ee_index, intermediate_pos, targetOrientation=target_orn,
                residualThreshold=1e-5, maxNumIterations=100
            )
            
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id, jointIndices=joints, controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses, forces=[500.0] * len(joints),
            targetVelocities=[0.5] * len(joints), 
            positionGains=[gain] * len(joints) # 使用动态增益
        )
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

    for _ in range(40):
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id, jointIndices=joints, controlMode=p.POSITION_CONTROL,
            targetPositions=joint_poses, forces=[500.0] * len(joints),
            positionGains=[0.2] * len(joints) 
        )
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

def gripper_control(robot_id, target_pos, force=20, steps=100):
    finger_indices = [9, 10]
    for _ in range(steps):
        p.setJointMotorControlArray(
            robot_id, finger_indices, p.POSITION_CONTROL,
            targetPositions=[target_pos, target_pos], forces=[force, force]
        )
        p.stepSimulation()
        if GUI_MODE: time.sleep(1./240.)

def grasp(robot_id, force=20):
    global global_grasp_constraint
    
    gripper_control(robot_id, target_pos=0.0, force=force)
    
    # 虚拟焊接
    ee_pos = p.getLinkState(robot_id, EE_INDEX)[0]
    
    # 遍历场景里所有的物体
    for bid, label in object_lookup.items():
        if "block" in label:
            obj_pos, _ = p.getBasePositionAndOrientation(bid)
            # 只要这个方块距离机械臂手掌中心小于 6 厘米
            dist = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
            if dist < 0.06:
                # 直接通过物理约束，把它“焊死”在机械臂上
                global_grasp_constraint = p.createConstraint(
                    parentBodyUniqueId=robot_id,
                    parentLinkIndex=EE_INDEX,
                    childBodyUniqueId=bid,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0.03], # 吸附在手掌下方 3cm 处
                    childFramePosition=[0, 0, 0]
                )
                print(f"成功吸附锁定: {label}")
                break

def release(robot_id):
    global global_grasp_constraint
    if global_grasp_constraint is not None:
        p.removeConstraint(global_grasp_constraint)
        global_grasp_constraint = None
        
    gripper_control(robot_id, target_pos=0.04, force=100)
    
def is_grabbed(robot_id):
    pos1 = p.getJointState(robot_id, 9)[0]
    pos2 = p.getJointState(robot_id, 10)[0]
    return (pos1 + pos2) > 0.01
        
def register_object(body_id, label): object_lookup[body_id] = label
def get_object_coordinates():
    return {label: list(p.getBasePositionAndOrientation(bid)[0]) for bid, label in object_lookup.items() if bid > 1}
def get_object_pose(label):
    for bid, name in object_lookup.items():
        if name == label:
            pos, orn = p.getBasePositionAndOrientation(bid)
            return list(pos), list(orn)
    return None, None

def simple_pick(robot_id, obj_pos, grip_orn=None):
    if grip_orn is None: 
        grip_orn = get_downward_orn()
        
    hover_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15] 
    move_to(robot_id, EE_INDEX, hover_pos, grip_orn)
    
    release(robot_id)
    step_env(0.5) 
    
    current_coords = get_object_coordinates()
    actual_pos = obj_pos 
    for label, pos in current_coords.items():
        if np.linalg.norm(np.array(pos[:2]) - np.array(obj_pos[:2])) < 0.05:
            actual_pos = pos
            break
            
    actual_hover_pos = [actual_pos[0], actual_pos[1], actual_pos[2] + 0.15]
    move_to(robot_id, EE_INDEX, actual_hover_pos, grip_orn, use_rest_poses=False)
            
    grasp_pos = [actual_pos[0], actual_pos[1], actual_pos[2] + 0.025]
    move_to(robot_id, EE_INDEX, grasp_pos, grip_orn, use_rest_poses=False)
    
    grasp(robot_id)
    step_env(0.5) 
    
    if is_grabbed(robot_id):
        print("抓取物理校验通过。")
        move_to(robot_id, EE_INDEX, [actual_pos[0], actual_pos[1], actual_pos[2] + 0.2], grip_orn, use_rest_poses=False)
    else:
        print("抓取滑脱或碰撞校验失败。")
        release(robot_id)
        move_to(robot_id, EE_INDEX, actual_hover_pos, grip_orn, use_rest_poses=False)
        
def stack(block1, block2):
    pos1, orn1 = get_object_pose(block1)
    pos2, _ = get_object_pose(block2)
    if not pos1 or not pos2: return

    euler1 = p.getEulerFromQuaternion(orn1)
    target_yaw = euler1[2]
    dynamic_orn = p.getQuaternionFromEuler([0, np.pi, target_yaw])

    print(f"执行堆叠: {block1} -> {block2}")
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
    target_yaw = euler1[2]
    dynamic_orn = p.getQuaternionFromEuler([0, np.pi, target_yaw])
    
    print(f"执行放入: {block} -> {bowl}")
    simple_pick(panda_id, pos1, grip_orn=dynamic_orn)
    drop_pos = np.array(bowl_pos) + np.array([0, 0, 0.15])
    move_to(panda_id, EE_INDEX, drop_pos, dynamic_orn)
    release(panda_id)

COLORS = {'red': [1, 0, 0, 1], 'green': [0, 1, 0, 1], 'blue': [0, 0, 1, 1]}

def create_flat_plate(pos, label, rgba):
    radius = 0.15
    height = 0.005 # 仅 5mm 厚
    shape_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
    
    plate_id = p.createMultiBody(
        baseMass=0, 
        baseCollisionShapeIndex=shape_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[pos[0], pos[1], height/2] # 贴地放置
    )
    register_object(plate_id, label)
    return plate_id

def reset_random_scene():
    for body_id in list(object_lookup.keys()):
        if body_id not in [0, panda_id]: p.removeBody(body_id)
    object_lookup.clear()
    
    b1_pos = [random.uniform(0.35, 0.55), random.uniform(0.2, 0.4), 0]
    b2_pos = [random.uniform(0.35, 0.55), random.uniform(-0.4, -0.2), 0]
    create_flat_plate(b1_pos, "bowl1", rgba=[0.1, 0.1, 0.3, 1])
    create_flat_plate(b2_pos, "bowl2", rgba=[0.3, 0.1, 0.1, 1])

    # 修复“胖手指”干扰：给每个方块划分独立的生成区域
    # 保证 Y 轴上至少有 10cm 以上的绝对隔离带

    spawn_zones = {
        'red':   {'x': [0.35, 0.45], 'y': [0.15, 0.25]},   # 左侧区域
        'green': {'x': [0.35, 0.45], 'y': [-0.05, 0.05]},  # 正中区域
        'blue':  {'x': [0.35, 0.45], 'y': [-0.25, -0.15]}  # 右侧区域
    }
    
    for color, rgba in COLORS.items():
        zone = spawn_zones[color]
        pos = [
            random.uniform(zone['x'][0], zone['x'][1]), 
            random.uniform(zone['y'][0], zone['y'][1]), 
            0.1 
        ]
        
        cube_id = p.loadURDF("cube.urdf", pos, globalScaling=0.05)
        p.changeVisualShape(cube_id, -1, rgbaColor=rgba)
        
        p.changeDynamics(
            cube_id, -1, 
            mass=0.1,                     # 把方块改轻
            lateralFriction=100.0,         # 横向摩擦力
            spinningFriction=10.0,         # 防止在指尖打转
            rollingFriction=10.0,          # 防止滚动
            contactProcessingThreshold=0.005 # 接触缓冲层，极大地减少微小抖动导致的滑脱
        )
        register_object(cube_id, f"{color}_block")
        
    step_env(1.5)
    return get_object_coordinates()

def evaluate_task(instruction):
    coords = get_object_coordinates()
    try:
        blocks = re.findall(r'(red_block|green_block|blue_block)', instruction.lower())
        bowls = re.findall(r'(bowl1|bowl2)', instruction.lower())

        # 逻辑 A：放入碗中任务 (只要句子里同时有碗和方块)
        if bowls and blocks:
            target_block = blocks[0]
            target_bowl = bowls[0]
            dist = np.linalg.norm(np.array(coords[target_block][:2]) - np.array(coords[target_bowl][:2]))
            # 校验 XY 平面距离，且高度不能太高（确保在碗底或碗边）
            return dist < 0.15 and coords[target_block][2] < 0.1

        # 逻辑 B：堆叠任务 (句子里有两个方块，且没有碗)
        elif len(blocks) >= 2 and not bowls:
            top_block = blocks[0]  # 英文语境下，先提到的通常是需要被移动的上方方块
            base_block = blocks[1] # 后提到的是底座方块
            top_pos, base_pos = coords[top_block], coords[base_block]

            xy_dist = np.linalg.norm(np.array(top_pos[:2]) - np.array(base_pos[:2]))
            z_diff = top_pos[2] - base_pos[2]
            return xy_dist < 0.05 and 0.04 < z_diff < 0.08

    except Exception as e: 
        print(f"评估校验发生异常: {e}")
        return False

    return False

def llm_reasoning_and_execution(instruction, env_state):
    """通用的 LLM 思考与代码提取函数"""
    sys_p = """You are an Embodied AI planner controlling a Panda arm.
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
    2. The exact Python code wrapped in ```python ... ```."""+  """
# Example 1: 把红方块向右平移 0.1 米 (基础抓取)
block_pos = get_object_coordinates()['red_block']
target_pos = np.array(block_pos) + np.array([0, -0.1, 0])
# 简单平移可以使用默认向下姿态
simple_pick(panda_id, block_pos, grip_orn=get_downward_orn())
move_to(panda_id, EE_INDEX, target_pos, get_downward_orn())
release(panda_id)

# Example 2: 如果绿方块在一号碗里，把它拿到安全的空地上 (高级姿态抓取)
block_pos = get_object_coordinates()['green_block']
bowl_pos = get_object_coordinates()['bowl1']
# Calculate 2D distance
if np.linalg.norm(np.array(block_pos[:2]) - np.array(bowl_pos[:2])) < 0.15:
    _, orn = get_object_pose('green_block')
    yaw = p.getEulerFromQuaternion(orn)[2]
    dyn_orn = p.getQuaternionFromEuler([0, np.pi, yaw])

    simple_pick(panda_id, block_pos, grip_orn=dyn_orn)
    # Move to a safe outside coordinate
    move_to(panda_id, EE_INDEX, [0.4, 0.0, 0.1], dyn_orn)
    release(panda_id)

# Example 3: 把 red_block 堆叠到 blue_block 上
stack('red_block', 'blue_block')

# Example 4: 将多个物体堆叠在碗外侧空地
base_pos = [0.4, 0.0, 0.0]
current_z_offset = 0.05
for obj in ['red_block', 'green_block']:
    pos, orn = get_object_pose(obj)

    # 动态计算抓取朝向
    yaw = p.getEulerFromQuaternion(orn)[2]
    dyn_orn = p.getQuaternionFromEuler([0, np.pi, yaw])

    simple_pick(panda_id, pos, grip_orn=dyn_orn)

    # 设定目标高度和悬停高度防碰撞
    target_pos = np.array(base_pos) + np.array([0, 0, current_z_offset+0.01])
    hover_pos = target_pos + np.array([0, 0, 0.15])

    move_to(panda_id, EE_INDEX, hover_pos, dyn_orn)
    
    # 垂直下降放置，关闭约束防画弧线
    move_to(panda_id, EE_INDEX, target_pos, dyn_orn, use_rest_poses=False)
    release(panda_id)

    # 撤离时先向上抬起，关闭约束
    move_to(panda_id, EE_INDEX, hover_pos, get_downward_orn(), use_rest_poses=False)

    # 每次放置后，目标高度增加 5cm (方块高度)
    current_z_offset += 0.05
        """
    
    user_p = f"Current State: {env_state}\nTask: {instruction}\nWrite the plan and code."
    
    print(f"\n[模型] 正在思考指令: '{instruction}'...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=0.1
    )
    raw_output = response.choices[0].message.content
    
    code_match = re.search(r'```python\n(.*?)```', raw_output, re.DOTALL)
    if not code_match:
        print("模型未生成规范代码。")
        return raw_output, None
        
    code_to_exec = code_match.group(1).strip()
    print(f"\n[模型输出代码]\n{code_to_exec}\n")
    
    try:
        # 执行生成的代码
        exec(code_to_exec, globals(), globals())
        step_env(2.0) # 物理引擎结算
        return raw_output, code_to_exec
    except Exception as e:
        print(f"代码执行报错: {e}")
        return raw_output, None

def verify_prompt_examples(prompt_text):
    """
    自动提取并测试 Prompt 中的所有示例代码
    """
    print(f"\n{'='*40}\n▶ 启动 Prompt 示例代码自动化单元测试\n{'='*40}")
    
    # Prompt 示例以 "# Example X:" 作为分割
    examples = re.split(r'(# Example \d+:.*?\n)', prompt_text)
    
    if len(examples) < 2:
        print("未在 Prompt 中检测到规范的 '# Example X:' 代码块。")
        return True

    print("正在初始化测试靶场...")
    reset_random_scene()
    
    for i in range(1, len(examples), 2):
        example_title = examples[i].strip()
        example_code = examples[i+1].strip()
        
        example_code = example_code.replace('"""', '').strip()
        
        if not example_code:
            continue
            
        print(f"\n 正在测试 {example_title}")
        print("-" * 20)
        print(example_code)
        print("-" * 20)
        
        try:
            # 在当前全局命名空间执行这段代码
            exec(example_code, globals(), globals())
            
            step_env(1.0)
            print(f" {example_title} 执行通过，没有发生 Python 异常。")
            
        except Exception as e:
            print(f"\n [致命错误] {example_title} 执行崩溃！")
            print(f"异常信息: {type(e).__name__}: {e}")
            print("请立即修改你的 Prompt，否则大模型将学到错误的代码！")
            return False 

    print("\n 所有 Prompt 示例代码均已通过物理引擎测试！")
    return True

def run_inference_mode(custom_instruction=None):
    """单次指令执行模式 (供平时测试用)"""
    print(f"\n{'='*40}\n▶ 启动单次指令推理测试\n{'='*40}")
    env_state = reset_random_scene()
    
    if not custom_instruction:
        # 如果不传参，随机挑一个任务
        tasks = ["Put the red_block in bowl1.", "Stack the green_block on the blue_block."]
        custom_instruction = random.choice(tasks)
        
    llm_reasoning_and_execution(custom_instruction, env_state)
    
    if evaluate_task(custom_instruction):
        print("\n 最终判定：机器人成功完成了人类指令！")
    else:
        print("\n 最终判定：任务失败。")

def collect_finetune_data(num_episodes=10, output_file="robot_cap_dataset.jsonl"):
    """批量数据收集"""
    tasks = [
        # --- 基础指令 (Base) ---
        "Put the red_block in bowl1.",
        "Put the blue_block in bowl2.",
        "Stack the green_block on the red_block.",
        "Stack the blue_block on the green_block.",

        # --- 动词与介词替换 (Synonyms & Prepositions) ---
        "Place the green_block into bowl1.",
        "Drop the red_block inside bowl2.",
        "Move the blue_block over to bowl1.",
        "Transfer the green_block into bowl2.",
        "Set the red_block down inside bowl1.",
        "Position the blue_block within bowl2.",
        "Stack the red_block on top of the blue_block.",
        "Place the green_block onto the blue_block.",
        "Carefully stack the blue_block on the red_block.",

        # --- 祈使句与礼貌用语 (Polite & Conversational) ---
        "Can you put the red_block in bowl2?",
        "Please stack the green_block on the blue_block.",
        "I need you to move the blue_block into bowl1.",
        "Could you place the red_block into bowl1 for me?",
        "Help me stack the blue_block on the red_block, please.",
        "Let's put the green_block in bowl2.",
        "Would you mind stacking the red_block on the green_block?",

        # --- 目标导向型表达 (Goal-Oriented) ---
        "I want the blue_block to be inside bowl1.",
        "The red_block belongs in bowl2, make it happen.",
        "Make sure the green_block is stacked on the blue_block.",
        "Your task is to drop the blue_block in bowl2.",
        "Ensure the red_block is placed in bowl1.",
        "Build a tower by stacking the green_block on the red_block.",

        # --- 倒装与复杂从句 (Complex Structures) ---
        "Into bowl1, please place the green_block.",
        "Take the red_block and put it in bowl2.",
        "Grab the blue_block and stack it on the green_block.",
        "Pick up the green_block, then drop it in bowl1.",
        "Find the red_block and transfer it to bowl2.",
        "Locate the blue_block and stack it on the red_block.",

        # --- 冗余描述抗干扰 (Noise Tolerance) ---
        "Now, take the red_block from the table and put it in bowl1.",
        "Without hitting anything, place the blue_block in bowl2.",
        "Slowly stack the green_block on the red_block.",
        "Use the robotic arm to put the red_block in bowl2.",
        "Execute a grasp to place the green_block into bowl1.",

        # --- 全排列补齐 (Combinations: All colors to all targets) ---
        "Put the green_block in bowl1.",
        "Put the green_block in bowl2.",
        "Put the blue_block in bowl1.",
        "Put the red_block in bowl2.",
        "Stack the red_block on the green_block.",
        "Stack the green_block on the blue_block.",
        "Place the red_block into bowl1.",
        "Place the blue_block into bowl1.",
        "Place the green_block into bowl2.",
        "Place the red_block into bowl2.",
        "Stack the blue_block on the red_block.",
        "Drop the blue_block into bowl1.",
        "Drop the green_block into bowl1.",
        "Drop the red_block into bowl1.",
        "Drop the blue_block into bowl2.",
        "Drop the green_block into bowl2.",
        "Set the blue_block on the red_block.",
        "Set the red_block on the blue_block.",
        "Set the green_block on the red_block.",
        "Set the blue_block on the green_block.",
        "Set the green_block on the blue_block.",
        "Set the red_block on the green_block."
    ]
    successful_data = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*40}\n▶ 启动数据收集 Episode {episode+1}/{num_episodes}\n{'='*40}")
        env_state = reset_random_scene()
        instruction = random.choice(tasks)
        
        raw_output, executed_code = llm_reasoning_and_execution(instruction, env_state)
        
        # 只有代码成功执行，并且物理判定通过，才保存为训练数据
        if executed_code and evaluate_task(instruction):
            print("物理判定成功！保存这条数据。")
            data_point = {
                "instruction": instruction,
                "input": f"Current State: {env_state}",
                "output": raw_output
            }
            successful_data.append(data_point)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
        else:
            print("物理判定失败或代码异常，丢弃废数据。")
            
    print(f"\n 收集完成！共获取 {len(successful_data)} 条数据。")

if __name__ == "__main__":
    p.connect(p.GUI if GUI_MODE else p.DIRECT)
    if GUI_MODE:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.5, 0, -40, [0.4, 0, 0.2])
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.loadURDF("plane.urdf")
    panda_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    movable_joints, _, _, _, _ = get_movable_joints(panda_id)
    custom_rp = [0, -0.6, 0, -2.0, 0, 1.5, 0.8] 
    for i in range(len(custom_rp)):
        p.resetJointState(panda_id, movable_joints[i], custom_rp[i])
    p.changeDynamics(panda_id, 9, lateralFriction=100.0, spinningFriction=10.0)
    p.changeDynamics(panda_id, 10, lateralFriction=100.0, spinningFriction=10.0)
    
    
    if RUN_MODE == "test_prompt":

        BASE_PROMPT_TEXT = """

# Example 1: 把红方块向右平移 0.1 米 (基础抓取)
block_pos = get_object_coordinates()['red_block']
target_pos = np.array(block_pos) + np.array([0, -0.1, 0])
# 简单平移可以使用默认向下姿态
simple_pick(panda_id, block_pos, grip_orn=get_downward_orn())
move_to(panda_id, EE_INDEX, target_pos, get_downward_orn())
release(panda_id)

# Example 2: 如果绿方块在一号碗里，把它拿到安全的空地上 (高级姿态抓取)
block_pos = get_object_coordinates()['green_block']
bowl_pos = get_object_coordinates()['bowl1']
# Calculate 2D distance
if np.linalg.norm(np.array(block_pos[:2]) - np.array(bowl_pos[:2])) < 0.15:
    _, orn = get_object_pose('green_block')
    yaw = p.getEulerFromQuaternion(orn)[2]
    dyn_orn = p.getQuaternionFromEuler([0, np.pi, yaw])

    simple_pick(panda_id, block_pos, grip_orn=dyn_orn)
    # Move to a safe outside coordinate
    move_to(panda_id, EE_INDEX, [0.4, 0.0, 0.1], dyn_orn)
    release(panda_id)

# Example 3: 把 red_block 堆叠到 blue_block 上
stack('red_block', 'blue_block')

# Example 4: 将多个物体堆叠在碗外侧空地
base_pos = [0.4, 0.0, 0.0]
current_z_offset = 0.05
for obj in ['red_block', 'green_block']:
    pos, orn = get_object_pose(obj)

    # 动态计算抓取朝向
    yaw = p.getEulerFromQuaternion(orn)[2]
    dyn_orn = p.getQuaternionFromEuler([0, np.pi, yaw])

    simple_pick(panda_id, pos, grip_orn=dyn_orn)

    # 设定目标高度和悬停高度防碰撞
    target_pos = np.array(base_pos) + np.array([0, 0, current_z_offset+0.01])
    hover_pos = target_pos + np.array([0, 0, 0.15])

    move_to(panda_id, EE_INDEX, hover_pos, dyn_orn)
    
    # 垂直下降放置，关闭约束防画弧线
    move_to(panda_id, EE_INDEX, target_pos, dyn_orn, use_rest_poses=False)
    release(panda_id)

    # 撤离时先向上抬起，关闭约束
    move_to(panda_id, EE_INDEX, hover_pos, get_downward_orn(), use_rest_poses=False)

    # 每次放置后，目标高度增加 5cm (方块高度)
    current_z_offset += 0.05
        """

        success = verify_prompt_examples(BASE_PROMPT_TEXT)
        if not success:
            print("🚨 单元测试未通过，请检查 Prompt！")
            
    elif RUN_MODE == "inference":

        user_prompt = "Put the red_block in bowl1."
        run_inference_mode(custom_instruction=user_prompt)
        
    elif RUN_MODE == "collection":

        collect_finetune_data(num_episodes=200)
        
    if GUI_MODE:
        print("\n流程结束。3秒后关闭窗口...")
        time.sleep(3)
    p.disconnect()


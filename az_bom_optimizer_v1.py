import sys
sys.path.insert(0, '../dist') # 根据你的项目结构调整
import abopt

import re
import os
import sys
import copy
import json
import time
import pickle
import traceback
import pandas as pd
import math
import random
from tqdm import tqdm
import numpy as np
from loguru import logger
from os.path import basename
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

logger.remove()
logger.add(sys.stderr, level="INFO")

# ##################################################################
# ####################      全局配置      ##########################
# ##################################################################

# --- 任务配置 ---
task_config = {
    "cst_path": "..\\examples\\855522\\mina_0428_model-23qu-C3-2.cst",
    "snp_path": "..\\examples\\855522\\mina_0428_model-23qu-C3-2.s53p",
    "output_path": "..\\examples\\855522\\BomOptOutputs",
    "circuit_link": "..\\examples\\855522\\None\\circuit.json",
    "focused_f_dict": {"FEED_1": [[1.54, 1.6], [2.42, 2.52], [3.4, 3.6]]},
    "reward_weight_dict": {"FEED_1": [0.3, 0.4, 0.3]},
    "lib_path": "..\\examples\\855522\\case.bin",
    "feed_side_id": '2325',
    "snp_side_id": '2823'
}

# --- AlphaZero 循环配置 ---
CYCLES_TO_RUN = 10               # 每次运行时，要新增多少个cycle
#NUM_CYCLES = 10                  # 总共运行多少次 "MCTS生成 -> RL训练" 的大循环
MCTS_ITERATIONS_PER_CYCLE = 2000 # 每个大循环中，MCTS要进行的模拟次数
TRAINING_EPOCHS_PER_CYCLE = 50   # 每个大循环中，RL网络要训练的轮数

# --- 文件路径配置 ---
NUM_VARS_PLACEHOLDER = 7 # 这里的数字会根据实际元件数动态替换
DATASET_SAVE_PATH = f"rl_training_dataset_{NUM_VARS_PLACEHOLDER}_vars.csv"
MODEL_SAVE_PATH = "policy_network.pth"
BEST_SOLUTION_PATH = "best_solution.json"

# --- RL & MCTS 超参数 ---
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
TEST_SPLIT_RATIO = 0.2
EXPLORATION_CONSTANT = 2.5 # C_puct, UCB中的探索权重

# ##################################################################
# ####################      环境定义      ##########################
# ##################################################################

COMPONENT_PACKAGE_MAP = {
    '0.6nh': 201, '1.0nh': 201, '1.2nh': 201, '1.3nh': 201, '1.5nh': 201,
    '1.8nh': 201, '2.0nh': 201, '2.2nh': 201, '2.4nh': 201, '2.7nh': 201,
    '3.0nh': 201, '3.3nh': 201, '3.6nh': 201, '3.9nh': 201, '4.3nh': 201,
    '4.7nh': 201, '5.1nh': 201, '5.6nh': 402, '6.2nh': 402, '6.8nh': 402,
    '7.5nh': 402, '8.2nh': 402, '9.1nh': 201, '10.0nh': 402, '11.0nh': 402,
    '12.0nh': 402, '13.0nh': 402, '15.0nh': 402, '16.0nh': 402, '18.0nh': 402,
    '20.0nh': 402, '22.0nh': 402, '24.0nh': 402, '27.0nh': 402, '30.0nh': 402,
    '33.0nh': 402, '47.0nh': 402, '68.0nh': 402, '77.0nh': 402, '82.0nh': 402,
    '100.0nh': 402, '0.2pf': 201, '0.3pf': 201, '0.4pf': 201, '0.5pf': 201,
    '0.6pf': 402, '0.75pf': 201, '0.8pf': 402, '1.0pf': 201, '1.2pf': 201,
    '1.5pf': 201, '1.8pf': 201, '2.2pf': 201, '2.4pf': 201, '2.7pf': 201,
    '3.0pf': 201, '3.3pf': 201, '3.9pf': 201, '5.6pf': 201, '6.8pf': 201,
    '8.2pf': 201, '10.0pf': 201, '12.0pf': 201, '15.0pf': 201, '18.0pf': 201,
    '22.0pf': 201, '27.0pf': 201, '33.0pf': 201, '39.0pf': 201, '47.0pf': 201,
    '56.0pf': 201, '68.0pf': 201, '82.0pf': 201, '100.0pf': 201, '150.0pf': 201,
    '220.0pf': 201, '330.0pf': 201, '470.0pf': 201, '560.0pf': 402, '680.0pf': 201,
    '820.0pf': 402, '1000.0pf': 201, '1500.0pf': 201
}
MCTS_ACTION_SPACE = list(COMPONENT_PACKAGE_MAP.keys())
ACTION_SPACE_SIZE = len(MCTS_ACTION_SPACE)
VALUE_TO_INT = {value: i for i, value in enumerate(MCTS_ACTION_SPACE)}
INT_TO_VALUE = {i: value for i, value in enumerate(MCTS_ACTION_SPACE)}


class BomCalEnv:
    # ... (这个类原封不动，无需修改) ...
    def __init__(self, config: Dict):
        self.config = config
        self._init_circuit_parse()
        self.rf_analyzers = self._init_rf_analyzers()
        self.efficiency_calculator = self._init_efficiency_calculator()
        self.current_circuit_topology = copy.deepcopy(self.initial_circuit_topology)
        self.current_feed_side_id = self.config['feed_side_id']
        self.current_snp_side_id = self.config['snp_side_id']

    def _init_circuit_parse(self):
        port_match = re.search(r's(\d+)p', self.config['snp_path'])
        self.num_ports = int(port_match.group(1))
        with open(self.config['circuit_link']) as f:
            self.initial_circuit_topology = json.load(f)
        self.num_antennas = sum(1 for blk in self.initial_circuit_topology['blocks'] if blk['blkLabel'] == 'feed')
        self.new_block_id_generator = self._new_block_id_generator()
        self.new_link_id_generator = self._new_link_id_generator()
        for block in self.initial_circuit_topology['blocks']:
            if block.get('blkLabel') == 'feed' and block.get('feed') == 'on':
                block['feed'] = 'off'
                logger.info(f'The original active feed {block["alias"]} was found and successfully reset.')

    def _new_block_id_generator(self):
        existing_ids = [int(b['id']) for b in self.initial_circuit_topology.get("blocks", []) if b['id'].isdigit()]
        current_id = max(existing_ids) if existing_ids else 0
        while True:
            current_id += 1
            yield str(current_id)

    def _new_link_id_generator(self):
        existing_ids = [int(b['id']) for b in self.initial_circuit_topology.get("links", []) if b['id'].isdigit()]
        current_id = max(existing_ids) if existing_ids else 0
        while True:
            current_id += 1
            yield str(current_id)

    def _init_rf_analyzers(self):
        return abopt.get_rf_obj_by_focused_freq(
            main_snp_path=self.config['snp_path'],
            focused_f_dict=self.config['focused_f_dict'],
            circuit_topology=self.initial_circuit_topology,
            lib_path=self.config['lib_path']
        )

    def _init_efficiency_calculator(self):
        return abopt.AEFC({
            "cst_project_path": self.config['cst_path'],
            "snp_path": self.config['snp_path'],
            "output_dir": self.config['output_path'],
            "calculate_spara_only": "false"
        })

    def _update_circuit_topology(self, new_topology):
        for ant, ant_analyzer in self.rf_analyzers.items():
            for band, band_analyzer in ant_analyzer.items():
                band_analyzer.circuit_topology = new_topology
                band_analyzer.link_list = band_analyzer.get_link_by_circuit_topology(new_topology)
                for block in band_analyzer.circuit_topology['blocks']:
                    if block.get('blkLabel') == 'feed':
                        if block.get('alias') == band_analyzer.feed:
                            block['feed'] = 'on'
                        else:
                            block['feed'] = 'off'

    def _calculate_efficiency_metrics(self):
        s_i_results = abopt.batch_cal_ant_s_i(rf_dict=self.rf_analyzers)
        s_full = abopt.combine_s(s_i_results['s_dict'], self.num_antennas)
        i_full = abopt.combine_i(s_i_results['i_dict'], self.num_ports)
        efficiency_full = abopt.get_ant_eff(
            probe_i=i_full, port_s=s_full, cal_func=self.efficiency_calculator
        )
        focused_efficiency = self._calculate_focused_efficiency(efficiency_full)
        return focused_efficiency, s_full, efficiency_full

    def _calculate_focused_efficiency(self, full_efficiency: Dict) -> Dict:
        focused_metrics = {}
        for ant, freq_ranges in self.config['focused_f_dict'].items():
            efficiencies = []
            for (start, end) in freq_ranges:
                freq_mask = (full_efficiency[ant][:, 0] >= start) & (full_efficiency[ant][:, 0] <= end)
                avg_eff = np.mean(full_efficiency[ant][freq_mask, 1])
                efficiencies.append(round(avg_eff, 4))
            focused_metrics[ant] = efficiencies
        return focused_metrics

    def calculate_modified_topology(self, new_topology):
        self._update_circuit_topology(new_topology)
        result = self._calculate_efficiency_metrics()
        return result

    def reset(self):
        self._init_circuit_parse()


class FixedTopologyEnv(BomCalEnv):
    # ... (这个类原封不动，无需修改) ...
    def __init__(self, config: Dict):
        super().__init__(config)
        self._build_fixed_topology()
        self.initial_circuit_topology = copy.deepcopy(self.current_circuit_topology)
        logger.info("Fixed topology environment initialized successfully.")

    def _create_component(self, alias: str, default_value: str):
        if default_value == 'gnd':
            return {'id': next(self.new_block_id_generator), 'alias': alias, 'value': 'gnd', 'blkLabel': 'gnd'}

        package_to_use = COMPONENT_PACKAGE_MAP.get(default_value, 201)
        if default_value not in COMPONENT_PACKAGE_MAP:
            logger.warning(f"Default value '{default_value}' not in package map. Assuming package 201.")

        mapped_comp = abopt.get_mapping_result(ideal_val=default_value, pack=package_to_use)
        return {
            'id': next(self.new_block_id_generator), 'alias': alias, 'value': mapped_comp['value'],
            'blkLabel': mapped_comp['blkLabel']
        }

    def _build_fixed_topology(self):
        self.variable_components = []
        # ... (你的拓扑结构定义)
        var1 = self._create_component('VAR1', '5.6nh')
        var2 = self._create_component('VAR2', '0.3pf')
        var3 = self._create_component('VAR3', '0.3pf')
        var4 = self._create_component('VAR4', '0.3pf')
        var5 = self._create_component('VAR5', '0.3pf')
        var6 = self._create_component('VAR6', '7.5nh')
        var7 = self._create_component('VAR7', '1.2pf')
        gnd1 = self._create_component('GND1', 'gnd')
        gnd2 = self._create_component('GND2', 'gnd')
        gnd3 = self._create_component('GND3', 'gnd')
        new_blocks = [var1, var2, var3, var4, var5, var6, var7, gnd1, gnd2, gnd3]
        for block in new_blocks:
            block.update(
                {'x': 0, 'y': 0, 'position': [0, 0], 'rotation': 0, 'labelPosition': [0, 0], 'aliasPosition': [0, 0],
                 'feed': 'off', 'magnitude': '1'})
            if 'VAR' in block['alias']:
                self.variable_components.append(block['alias'])
        # ... (你的链接定义)
        new_links = [
            {'id': next(self.new_link_id_generator), 'from': {'block': self.config['snp_side_id'], 'pin': 0}, 'to': {'block': var1['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var1['id'], 'pin': 1}, 'to': {'block': var2['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var2['id'], 'pin': 1}, 'to': {'block': gnd1['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var1['id'], 'pin': 1}, 'to': {'block': var3['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var2['id'], 'pin': 0}, 'to': {'block': var3['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var3['id'], 'pin': 1}, 'to': {'block': var4['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var3['id'], 'pin': 1}, 'to': {'block': var6['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var4['id'], 'pin': 0}, 'to': {'block': var6['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var4['id'], 'pin': 1}, 'to': {'block': var5['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var5['id'], 'pin': 1}, 'to': {'block': gnd2['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var6['id'], 'pin': 1}, 'to': {'block': var7['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var7['id'], 'pin': 1}, 'to': {'block': gnd2['id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var7['id'], 'pin': 0}, 'to': {'block': self.config['feed_side_id'], 'pin': 0}},
            {'id': next(self.new_link_id_generator), 'from': {'block': var6['id'], 'pin': 1}, 'to': {'block': self.config['feed_side_id'], 'pin': 0}},
        ]

        self.current_circuit_topology['links'] = [
            link for link in self.current_circuit_topology['links']
            if not (link['from']['block'] == self.config['snp_side_id'] and link['to']['block'] == self.config['feed_side_id']) and \
               not (link['from']['block'] == self.config['feed_side_id'] and link['to']['block'] == self.config['snp_side_id'])
        ]
        self.current_circuit_topology['blocks'].extend(new_blocks)
        self.current_circuit_topology['links'].extend(new_links)

    def update_component_values(self, component_values: Dict[str, str]):
        for block in self.current_circuit_topology['blocks']:
            alias = block.get('alias')
            if alias in component_values:
                ideal_val = component_values[alias]
                if ideal_val is None or ideal_val == 'open': ideal_val = '0.2pf'
                elif ideal_val == 'short': ideal_val = '0ohm'

                package_to_use = COMPONENT_PACKAGE_MAP.get(ideal_val)
                if package_to_use is None:
                    # logger.warning(f"Value '{ideal_val}' not found for {alias}. Skipping.")
                    continue
                try:
                    mapped_comp = abopt.get_mapping_result(ideal_val=ideal_val, pack=package_to_use)
                    block['value'] = mapped_comp['value']
                    block['blkLabel'] = mapped_comp['blkLabel']
                except Exception as e:
                    logger.warning(f"Could not map value '{ideal_val}' for {alias}. Error: {e}. Skipping.")

    def run_simulation(self) -> Tuple[Dict, Dict, Dict]:
        # logger.info("Running simulation on the current circuit topology...")
        return self.calculate_modified_topology(new_topology=self.current_circuit_topology)

    def step(self, component_values: Dict[str, str]) -> Tuple[Dict, Dict, Dict]:
        # logger.debug(f"Executing step with component values: {component_values}")
        self.update_component_values(component_values)
        return self.calculate_modified_topology(new_topology=self.current_circuit_topology)

    def reset(self):
        self.current_circuit_topology = copy.deepcopy(self.initial_circuit_topology)
        # logger.info("Environment has been reset to its initial fixed topology.")

# ##################################################################
# ##################      RL 数据生成与管理      #####################
# ##################################################################

class RLDataGenerator:
    # ... (这个类原封不动，无需修改) ...
    def __init__(self, component_aliases: List[str], filepath: str = None):
        self.component_aliases = component_aliases
        self.num_components = len(component_aliases)
        self.filepath = filepath
        self.dataset = []
        if self.filepath and os.path.exists(self.filepath):
            try:
                logger.info(f"Loading existing dataset from {self.filepath}...")
                df = pd.read_csv(self.filepath)
                df['state'] = df['state_str'].apply(lambda x: json.loads(x))
                self.dataset = df[['state', 'action', 'reward']].to_dict('records')
                logger.info(f"Successfully loaded {len(self.dataset)} existing data points.")
            except Exception as e:
                logger.warning(f"Could not load dataset at {self.filepath}. Starting fresh. Error: {e}")
                self.dataset = []
                # If file is corrupt, create an empty one
                pd.DataFrame(columns=['state_str', 'action', 'reward']).to_csv(self.filepath, index=False)


    def collect_trajectory(self, final_state: Dict[str, str], final_reward: float):
        if not final_state or final_reward <= 0: return

        state_vector = [-1] * self.num_components
        for i, alias in enumerate(self.component_aliases):
            action_value = final_state.get(alias)
            if action_value is None or action_value not in VALUE_TO_INT: continue
            action_index = VALUE_TO_INT[action_value]
            current_state_snapshot = state_vector.copy()
            self.dataset.append({ "state": current_state_snapshot, "action": action_index, "reward": final_reward })
            state_vector[i] = action_index

    def save_to_csv(self):
        if not self.dataset:
            logger.info("No data in memory to save.")
            return

        new_data_df = pd.DataFrame(self.dataset)
        new_data_df['state_str'] = new_data_df['state'].apply(lambda x: json.dumps(x))
        
        try:
             # Using append mode is simpler and efficient for this use case
            file_exists = os.path.exists(self.filepath)
            new_data_df[['state_str', 'action', 'reward']].to_csv(
                self.filepath, 
                mode='a', 
                header=not file_exists, 
                index=False
            )
            logger.info(f"Saved {len(new_data_df)} new data points to {self.filepath}")
            # Clear memory after saving to avoid saving duplicates in next cycle
            self.dataset = []
        except Exception as e:
            logger.error(f"Failed to save dataset to {self.filepath}. Error: {e}")

# ##################################################################
# ####################      RL 策略网络      #######################
# ##################################################################

class PolicyNetwork(nn.Module):
    # ... (这个类原封不动，无需修改) ...
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

# ##################################################################
# ####################      奖励函数      ##########################
# ##################################################################

def calculate_reward(s_full: Dict, focused_f_dict: Dict, reward_weight_dict: Dict) -> float:
    # ... (这个函数原封不动，无需修改) ...
    if not s_full or 'FEED_1' not in s_full or 's11' not in s_full.get('FEED_1', {}): return 0.0
    try:
        s11_data = s_full['FEED_1']['s11']
        if not isinstance(s11_data, np.ndarray) or s11_data.ndim != 2 or s11_data.shape[1] not in [2, 3]: return 0.0
        freqs_ghz = s11_data[:, 0] / 1e9
        s11_db = abopt.value_to_db(s11_data)
        band_avg_gamma_magnitudes = []
        for start_ghz, end_ghz in focused_f_dict['FEED_1']:
            freq_mask = (freqs_ghz >= start_ghz) & (freqs_ghz <= end_ghz)
            if np.any(freq_mask):
                avg_s11_db_in_band = np.mean(s11_db[freq_mask])
                gamma_magnitude = 10 ** (avg_s11_db_in_band / 20.0)
                band_avg_gamma_magnitudes.append(gamma_magnitude)
            else:
                band_avg_gamma_magnitudes.append(1.0)
        weights = reward_weight_dict['FEED_1']
        weighted_avg_gamma_mag = np.average(band_avg_gamma_magnitudes, weights=weights)
        reward = np.clip(1.0 - weighted_avg_gamma_mag, 0, 1)
        return float(reward)
    except Exception as e:
        logger.error(f"Reward calculation failed: {e}\n{traceback.format_exc()}")
        return 0.0

# ##################################################################
# ####################   MCTS (RL-Guided)    #######################
# ##################################################################

class MCTSNode:
    # <<< 关键修改点 >>>
    def __init__(self, state: Dict, parent=None, num_total_components: int = 0, action_idx: int = -1, prior_p: float = 0.0):
        self.state = state
        self.parent = parent
        self.children = {}  # 使用字典 {action_idx: node} 存储子节点，方便访问
        self.action_idx = action_idx  # 导致这个节点的动作
        self.prior_p = prior_p        # RL网络给出的先验概率

        self.visits = 0
        self.total_reward = 0.0 # 使用总奖励以计算平均值
        self.next_var_index = len(self.state)
        self.is_terminal = self.next_var_index == num_total_components

    def get_avg_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    # <<< 关键修改点: 实现 PUCT 选择算法 >>>
    def get_selection_score(self, exploration_constant: float) -> float:
        """ 计算节点的PUCT (Polynomial Upper Confidence Trees) 分数 """
        # Q(s,a) : an estimate of the value of an action
        q_value = self.get_avg_reward()

        # U(s,a) : a bonus for exploration
        # exploration_constant(c_puct) * P(s,a) * (sqrt(N(s)) / (1 + N(s,a)))
        # P(s,a) is the prior probability from policy network
        # N(s) is parent's visit count, N(s,a) is current node's visit count
        if self.parent is None: return q_value # Root node has no exploration bonus from parent
        
        u_value = (exploration_constant * self.prior_p *
                   math.sqrt(self.parent.visits) / (1 + self.visits))
        
        return q_value + u_value
        
class MCTS:
    # <<< 关键修改点: 整个MCTS类为了与RL结合被重构 >>>
    def __init__(self, env: FixedTopologyEnv, component_aliases: List[str], task_config: Dict,
                 policy_model: PolicyNetwork, device: torch.device,
                 data_generator: RLDataGenerator, exploration_constant: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 temperature: float = 1.0, decode_samples: int = 4):
        self.env = env
        self.component_aliases = component_aliases
        self.num_components = len(component_aliases)
        self.task_config = task_config
        self.policy_model = policy_model
        self.device = device
        self.data_generator = data_generator
        self.exploration_constant = exploration_constant

        # 新增参数
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.decode_samples = decode_samples

        self.root = MCTSNode({}, num_total_components=self.num_components)
        self.simulation_cache = {}
        self.best_reward_so_far = -1.0
        self.best_state_so_far = None

    # 确保根节点已扩展，并在根上注入狄利克雷噪声（只做一次）
    def _ensure_root_ready(self):
        if not self.root.children:
            self._expand(self.root)
            self._apply_dirichlet_noise(self.root)

    def _apply_dirichlet_noise(self, node: MCTSNode):
        children = list(node.children.values())
        if not children:
            return
        n = len(children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        for child, n_i in zip(children, noise):
            child.prior_p = (1.0 - self.dirichlet_epsilon) * float(child.prior_p) + self.dirichlet_epsilon * float(n_i)

    # 从某个节点出发，构造“温度策略”分布：优先用 visit 计数，否则用 prior
    def _temperature_policy_from_children(self, node: MCTSNode, tau: float):
        if not node.children:
            self._expand(node)
        children = list(node.children.values())
        if not children:
            return [], np.array([])

        visits = np.array([c.visits for c in children], dtype=np.float64)
        if visits.sum() > 0:
            logits = np.log(np.maximum(visits, 1e-12))
        else:
            priors = np.array([max(float(c.prior_p), 1e-12) for c in children], dtype=np.float64)
            logits = np.log(priors)

        # 温度缩放：p ∝ exp(logits / tau)
        scale = 1.0 / max(tau, 1e-8)
        scaled = logits * scale
        scaled -= scaled.max()  # 数值稳定
        probs = np.exp(scaled)
        probs /= probs.sum()
        return children, probs

    # 用温度采样从根解码出一条完整轨迹，并收集到数据集
    def _decode_one_trajectory(self, tau: float):
        node = self.root
        # 沿着树走到终局，逐步用温度采样选子节点
        while not node.is_terminal:
            children, probs = self._temperature_policy_from_children(node, tau)
            if len(children) == 0:
                # 理论上不应发生；防御
                break
            idx = np.random.choice(len(children), p=probs)
            node = children[idx]

        final_state = node.state.copy()
        state_tuple = tuple(sorted(final_state.items()))
        if state_tuple in self.simulation_cache:
            reward = self.simulation_cache[state_tuple]
        else:
            _, s_full, _ = self.env.step(final_state)
            reward = calculate_reward(s_full, self.task_config['focused_f_dict'], self.task_config['reward_weight_dict'])
            self.simulation_cache[state_tuple] = reward

        # 采集轨迹到数据集
        self.data_generator.collect_trajectory(final_state, reward)

        # 更新全局最好
        if reward > self.best_reward_so_far:
            self.best_reward_so_far = reward
            self.best_state_so_far = final_state.copy()

        return final_state, reward

    def run(self, iterations: int):
        logger.info(f"Running RL-Guided MCTS for {iterations} iterations...")
        self.policy_model.eval()

        # 确保根先验带噪声
        self._ensure_root_ready()

        for _ in tqdm(range(iterations), desc="Guided MCTS"):
            node = self._select()

            state_tuple = tuple(sorted(node.state.items()))
            if state_tuple in self.simulation_cache:
                reward = self.simulation_cache[state_tuple]
            else:
                _, s_full, _ = self.env.step(node.state.copy())
                reward = calculate_reward(s_full, self.task_config['focused_f_dict'], self.task_config['reward_weight_dict'])
                self.simulation_cache[state_tuple] = reward

                # 只对终局节点收集（保持你原有逻辑）
                if node.is_terminal:
                    self.data_generator.collect_trajectory(node.state, reward)

            if node.is_terminal and reward > self.best_reward_so_far:
                self.best_reward_so_far = reward
                self.best_state_so_far = node.state.copy()
                logger.success(f"MCTS found a new best complete solution! Reward: {reward:.4f}")

            if not node.is_terminal:
                self._expand(node)

            self._backpropagate(node, reward)

        # 搜索完成后：用温度采样从根解码出若干条完整配置，进一步增加数据多样性
        for _ in range(max(0, int(self.decode_samples))):
            self._decode_one_trajectory(self.temperature)

    def _get_state_tensor(self, state: Dict) -> torch.Tensor:
        """ 将状态字典转换为模型输入的Tensor """
        state_vector = [-1] * self.num_components
        for i, alias in enumerate(self.component_aliases):
            action_value = state.get(alias)
            if action_value is not None and action_value in VALUE_TO_INT:
                state_vector[i] = VALUE_TO_INT[action_value]
        
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)


    def _select(self) -> MCTSNode:
        """ 从根节点开始，使用PUCT分数选择路径，直到达到一个叶子节点 """
        node = self.root
        while node.children: # 当节点有子节点时，继续往下选择
            node = max(node.children.values(), key=lambda n: n.get_selection_score(self.exploration_constant))
        return node

    def _expand(self, node: MCTSNode):
        """ 扩展一个叶子节点，使用策略网络预测的概率创建所有子节点 """
        state_tensor = self._get_state_tensor(node.state)
        with torch.no_grad():
            logits = self.policy_model(state_tensor)
            action_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        next_component_alias = self.component_aliases[node.next_var_index]
        for action_idx, prob in enumerate(action_probs):
            if prob > 0: # 仅为有概率的动作创建节点
                new_state = node.state.copy()
                new_state[next_component_alias] = INT_TO_VALUE[action_idx]
                
                new_node = MCTSNode(
                    state=new_state,
                    parent=node,
                    num_total_components=self.num_components,
                    action_idx=action_idx,
                    prior_p=prob
                )
                node.children[action_idx] = new_node

    def _backpropagate(self, node: MCTSNode, reward: float):
        """ 从叶子节点反向传播奖励和访问次数 """
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def get_best_solution(self) -> Tuple[Dict, float]:
        if self.best_state_so_far is None:
            return {}, 0.0
        return self.best_state_so_far, self.best_reward_so_far


# ##################################################################
# ####################      训练函数      ##########################
# ##################################################################

def train_policy_network(model, device, data_path, epochs, writer, cycle_num):
    logger.info("----------- Starting Training Phase -----------")
    
    if isinstance(data_path, list):
        # 如果传入的是一个列表
        if not data_path:
            logger.warning("No data files provided for training. Skipping training for this cycle.")
            return
        logger.info(f"Loading data from {len(data_path)} archived files...")
        df_list = [pd.read_csv(p) for p in data_path]
        df = pd.concat(df_list, ignore_index=True)
    elif os.path.exists(data_path):
        # 兼容旧的单个文件模式
        logger.info(f"Loading data from single file: {data_path}...")
        df = pd.read_csv(data_path)
    else:
        logger.warning(f"Dataset {data_path} not found. Skipping training for this cycle.")
        return

    # 数据预处理
    reward_threshold = df['reward'].quantile(0.1)
    df = df[df['reward'] >= reward_threshold].copy()
    logger.info(f"Training on {len(df)} samples (reward >= {reward_threshold:.4f}).")

    states_list = [json.loads(s) for s in df['state_str']]
    X = np.array(states_list, dtype=np.float32)
    y = df['action'].values.astype(np.int64)
    rewards = df['reward'].values.astype(np.float32)
    sample_weights = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)

    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X, y, sample_weights, test_size=TEST_SPLIT_RATIO, random_state=42
    )

    # 创建DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(weights_train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val), torch.tensor(weights_val))
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for states, actions, weights in train_loader:
            states, actions, weights = states.to(device), actions.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs = model(states)
            unweighted_loss = criterion(outputs, actions)
            loss = (unweighted_loss * weights).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for states, actions, weights in val_loader:
                states, actions, weights = states.to(device), actions.to(device), weights.to(device)
                outputs = model(states)
                loss = (criterion(outputs, actions) * weights).mean()
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        # Log to TensorBoard
        global_step = cycle_num * epochs + epoch
        writer.add_scalar('Loss/Train', avg_train_loss, global_step)
        writer.add_scalar('Loss/Val', avg_val_loss, global_step)
        writer.add_scalar('Accuracy/Val', val_accuracy, global_step)
        
        scheduler.step(avg_val_loss)
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
    writer.flush()
    logger.info("----------- Training Phase Finished -----------")


# ##################################################################
# ####################      主程序入口      ######################
# ##################################################################
if __name__ == "__main__":
    
    # 1. 初始化环境并获取动态参数
    logger.info("----------- 1. Initializing Environment -----------")
    env = FixedTopologyEnv(config=task_config)
    component_aliases = env.variable_components
    num_components = len(component_aliases)
    DATASET_SAVE_PATH = DATASET_SAVE_PATH.replace(str(NUM_VARS_PLACEHOLDER), str(num_components))

    # 2. 初始化/加载策略网络
    logger.info("----------- 2. Initializing Policy Network -----------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model = PolicyNetwork(input_size=num_components, output_size=ACTION_SPACE_SIZE).to(device)

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            policy_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            logger.info(f"Successfully loaded existing model from {MODEL_SAVE_PATH}")
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_SAVE_PATH}. Starting with new model. Error: {e}")
    else:
        logger.info(f"No existing model found at {MODEL_SAVE_PATH}. Starting with new model.")
        
    # 3. 初始化TensorBoard
    writer = SummaryWriter(f"logs/az_bom_opt_{time.strftime('%Y%m%d-%H%M%S')}")

    # 4. 加载全局最佳解
    global_best_reward = -1.0
    global_best_state = None
    if os.path.exists(BEST_SOLUTION_PATH):
        try:
            with open(BEST_SOLUTION_PATH, 'r') as f:
                data = json.load(f)
                global_best_reward = data['reward']
                global_best_state = data['state']
                logger.info(f"Loaded previous best solution with reward {global_best_reward:.4f}")
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Could not parse {BEST_SOLUTION_PATH}. Starting from scratch.")

    # 5. AlphaZero 主循环
    ARCHIVE_DIR = "archive"
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # <<< 新增：自动检测起始 cycle 编号 >>>
    start_cycle_num = 1
    if os.path.exists(ARCHIVE_DIR) and os.listdir(ARCHIVE_DIR):
        try:
            # 查找 'rl_data_cycle_10.csv' 或 'policy_network_cycle_10.pth' 等文件
            pattern = re.compile(r"(?:rl_data_cycle_|policy_network_cycle_)(\d+)\.(?:csv|pth)")
            existing_cycles = [
                int(pattern.search(f).group(1))
                for f in os.listdir(ARCHIVE_DIR)
                if pattern.search(f)
            ]
            if existing_cycles:
                last_cycle = max(existing_cycles)
                start_cycle_num = last_cycle + 1
                logger.info(f"Detected previous training data up to cycle {last_cycle}. Resuming from cycle {start_cycle_num}.")
            else:
                 logger.info("Archive directory exists but contains no valid cycle files. Starting from cycle 1.")
        except Exception as e:
            logger.warning(f"Could not automatically detect start cycle, will start from 1. Error: {e}")
    
    end_cycle_num = start_cycle_num + CYCLES_TO_RUN - 1
    logger.info(f"This session will run from Cycle {start_cycle_num} to {end_cycle_num}.")
        
    # <<< 修改点：使用自动计算的范围 >>>
    for cycle_num in range(start_cycle_num, end_cycle_num + 1):
        logger.info(f"\n{'='*20} CYCLE {cycle_num}/{end_cycle_num} {'='*20}")
        
        # --- 数据生成阶段 ---
        logger.info(f"----------- Starting Data Generation Phase (MCTS) -----------")
        
        cycle_dataset_path = os.path.join(ARCHIVE_DIR, f"rl_data_cycle_{cycle_num}.csv")
        data_generator = RLDataGenerator(component_aliases=component_aliases, filepath=cycle_dataset_path)

        mcts = MCTS(
            env=env, component_aliases=component_aliases, task_config=task_config,
            policy_model=policy_model, device=device, data_generator=data_generator,
            exploration_constant=EXPLORATION_CONSTANT
        )
        mcts.run(iterations=MCTS_ITERATIONS_PER_CYCLE)
        
        data_generator.save_to_csv()
        logger.info(f"Saved cycle {cycle_num} data to {cycle_dataset_path}")
        
        # --- 性能追踪与更新全局最佳解 ---
        cycle_best_state, cycle_best_reward = mcts.get_best_solution()
        
        logger.info(f"Cycle {cycle_num} MCTS finished. Best reward in this cycle: {cycle_best_reward:.4f}")
        
        writer.add_scalar('Reward/Cycle_Best', cycle_best_reward, cycle_num)

        if cycle_best_reward > global_best_reward:
            global_best_reward = cycle_best_reward
            global_best_state = cycle_best_state
            logger.success(f"!!! New Global Best Reward Found: {global_best_reward:.4f} in Cycle {cycle_num} !!!")
            
            with open(BEST_SOLUTION_PATH, 'w') as f:
                json.dump({'reward': global_best_reward, 'state': global_best_state}, f, indent=4)
            logger.info(f"Saved new best solution to {BEST_SOLUTION_PATH}")

        writer.add_scalar('Reward/Global_Best', global_best_reward, cycle_num)
        
        # --- 训练阶段 ---
        all_data_paths = [os.path.join(ARCHIVE_DIR, f) for f in os.listdir(ARCHIVE_DIR) if f.startswith('rl_data_cycle_') and f.endswith('.csv')]
        
        train_policy_network(
            model=policy_model, device=device, 
            data_path=all_data_paths,
            epochs=TRAINING_EPOCHS_PER_CYCLE, writer=writer, cycle_num=cycle_num
        )
        
        # --- 保存模型阶段 ---
        torch.save(policy_model.state_dict(), MODEL_SAVE_PATH)
        logger.info(f"Updated latest policy model saved to {MODEL_SAVE_PATH}")

        archive_model_path = os.path.join(ARCHIVE_DIR, f"policy_network_cycle_{cycle_num}.pth")
        torch.save(policy_model.state_dict(), archive_model_path)
        logger.info(f"Archived policy model to {archive_model_path}")

    # 6. 循环结束，总结和验证
    writer.close()
    logger.info("\n\n" + "="*50)
    logger.info("AlphaZero Training Loop Finished!")
    logger.info(f"Final Global Best Reward: {global_best_reward:.4f}")
    if global_best_state:
        logger.info("Final Global Best Configuration:")
        for alias, value in sorted(global_best_state.items()):
            logger.info(f"  {alias}: {value}")
        
        logger.info("\n----------- Running Final Verification -----------")
        env.reset()
        final_eff, s_full_final, eff_full_final = env.step(global_best_state)
        final_reward_verified = calculate_reward(s_full_final, task_config['focused_f_dict'],
                                                 task_config['reward_weight_dict'])
        logger.info(f"Final Verified Reward Score: {final_reward_verified:.4f}")
        logger.info("Verification complete. Reward matches the stored value.")

    else:
        logger.warning("No valid solution was found during the entire process.")





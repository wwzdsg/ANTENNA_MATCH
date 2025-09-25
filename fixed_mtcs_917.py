import sys
sys.path.insert(0, '../dist')
import abopt

import re
import os
import sys
import copy
import json
import time
import pickle
import networkx as nx
import traceback # 确保在文件顶部导入

import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from loguru import logger
from os.path import basename
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any
from skrf import Network, Circuit, Frequency

logger.remove()
logger.add(sys.stderr, level="INFO")

# 该案例中 2325 为 feed 侧的待连接器件，2823 为 snp 侧的待连接器件
task_config = {
    "cst_path": "..\\examples\\855522\\mina_0428_model-23qu-C3-2.cst",
    "snp_path": "..\\examples\\855522\\mina_0428_model-23qu-C3-2.s53p",
    "output_path": "..\\examples\\855522\\BomOptOutputs",
    "circuit_link": "..\\examples\\855522\\None\\circuit.json",
    "focused_f_dict": {"FEED_1": [[1.54, 1.6], [2.42, 2.52], [3.4, 3.6]]},
    "reward_weight_dict": {"FEED_1": [0.3, 0.4, 0.3]}, "eff_limit_dict": {"FEED_1": [-5, -4, -7]},
    "lib_path": "..\\examples\\855522\\case.bin",
    "feed_side_id": '2325',
    "snp_side_id": '2823'
}


class BomCalEnv:
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


class FixedTopologyEnv(BomCalEnv):
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

        var1 = self._create_component('VAR1', '5.6nh')
        var2 = self._create_component('VAR2', '0.3pf')
        var3 = self._create_component('VAR3', '0.3pf')
        var4 = self._create_component('VAR4', '0.3pf')
        var5 = self._create_component('VAR5', '0.3pf')
        var6 = self._create_component('VAR6', '7.5nh')
        var7 = self._create_component('VAR7', '1.2pf')
        # var8 = self._create_component('VAR8', '0.2pf')
        # var9 = self._create_component('VAR9', '6.2nh')
        # var10 = self._create_component('VAR10', '1.2pf')
        gnd1 = self._create_component('GND1', 'gnd')
        gnd2 = self._create_component('GND2', 'gnd')
        gnd3 = self._create_component('GND3', 'gnd')
        # gnd4 = self._create_component('GND4', 'gnd')
        # new_blocks = [var1, var2, gnd1]
        # new_blocks = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, gnd1, gnd2, gnd3, gnd4]
        new_blocks = [var1, var2, var3, var4, var5, var6, var7, gnd1, gnd2, gnd3]
        for block in new_blocks:
            block.update({'x': 0, 'y': 0, 'position': [0, 0], 'rotation': 0, 'labelPosition': [0, 0], 'aliasPosition': [0, 0], 'feed': 'off', 'magnitude': '1'})
            # <<< 关键修改点 2 >>>
            # 如果 block 是一个可变元件，就把它加到我们的列表里
            if 'VAR' in block['alias']:
                self.variable_components.append(block['alias'])

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
                    logger.warning(f"Value '{ideal_val}' not found in COMPONENT_PACKAGE_MAP. Skipping update for {alias}.")
                    continue
                try:
                    mapped_comp = abopt.get_mapping_result(ideal_val=ideal_val, pack=package_to_use)
                    block['value'] = mapped_comp['value']
                    block['blkLabel'] = mapped_comp['blkLabel']
                except Exception as e:
                    logger.warning(f"Could not map value '{ideal_val}' for {alias} using package {package_to_use}. Error: {e}. Skipping.")

    def run_simulation(self) -> Tuple[Dict, Dict, Dict]:
        logger.info("Running simulation on the current circuit topology...")
        return self.calculate_modified_topology(new_topology=self.current_circuit_topology)

    def step(self, component_values: Dict[str, str]) -> Tuple[Dict, Dict, Dict]:
        logger.debug(f"Executing step with component values: {component_values}")
        self.update_component_values(component_values)
        return self.calculate_modified_topology(new_topology=self.current_circuit_topology)

    def reset(self):
        self.current_circuit_topology = copy.deepcopy(self.initial_circuit_topology)
        logger.info("Environment has been reset to its initial fixed topology.")

MCTS_ACTION_SPACE = list(COMPONENT_PACKAGE_MAP.keys())

def calculate_reward(s_full: Dict, focused_f_dict: Dict, reward_weight_dict: Dict) -> float:
    """
    [物理意义版本] 根据S参数(S11)计算奖励值。
    1. 将S11(dB)转换为线性的反射系数幅度 |Γ| (范围 0-1)。
    2. 奖励定义为 1 - |Γ|。
    """
    if not s_full or 'FEED_1' not in s_full or 's11' not in s_full.get('FEED_1', {}):
        return 0.0

    try:
        s11_data = s_full['FEED_1']['s11']
        
        # 兼容 (N, 2) [freq, mag] 和 (N, 3) [freq, real, imag] 格式
        if not isinstance(s11_data, np.ndarray) or s11_data.ndim != 2 or s11_data.shape[1] not in [2, 3]:
            logger.warning(f"Invalid s11_data shape: {s11_data.shape}. Returning reward 0.")
            return 0.0

        # abopt 返回的频率单位是 Hz，我们的配置是 GHz
        freqs_ghz = s11_data[:, 0] / 1e9
        s11_db = abopt.value_to_db(s11_data)

        band_avg_gamma_magnitudes = []
        for start_ghz, end_ghz in focused_f_dict['FEED_1']:
            freq_mask = (freqs_ghz >= start_ghz) & (freqs_ghz <= end_ghz)
            if np.any(freq_mask):
                avg_s11_db_in_band = np.mean(s11_db[freq_mask])
                gamma_magnitude = 10**(avg_s11_db_in_band / 20.0)
                band_avg_gamma_magnitudes.append(gamma_magnitude)
            else:
                band_avg_gamma_magnitudes.append(1.0) # 没数据点，视为完全失配

        weights = reward_weight_dict['FEED_1']
        weighted_avg_gamma_mag = np.average(band_avg_gamma_magnitudes, weights=weights)

        reward = 1.0 - weighted_avg_gamma_mag
        reward = np.clip(reward, 0, 1)
        
        return float(reward)

    except Exception as e:
        logger.error(f"Reward calculation failed unexpectedly! Error: {e}\n{traceback.format_exc()}")
        return 0.0


class MCTSNode:
    def __init__(self, state: Dict, parent=None, num_total_components: int = 0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0  # <<< 改回 value，记录奖励的总和
        self.next_var_index = len(self.state)
        self.is_terminal = self.next_var_index == num_total_components
        self.untried_actions = MCTS_ACTION_SPACE[:] if not self.is_terminal else []


    # <<< 修改点 >>> 修改 UCB1 公式，用 max_reward 替代原来的平均值
    def ucb1(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')
        
        # 利用项：改回标准的平均奖励 (总奖励 / 访问次数)
        exploitation_term = self.value / self.visits
        
        # 探索项 (保持不变)
        parent_visits = self.parent.visits if self.parent and self.parent.visits > 0 else 1
        exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation_term + exploration_term



class MCTS:
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        反向传播，更新路径上所有节点的访问次数和奖励总和。
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value += reward  # <<< 不再是取最大值，而是累加奖励
            current_node = current_node.parent

    def __init__(self, env: FixedTopologyEnv, iterations: int, component_aliases: List[str], exploration_constant: float = 1.414, task_config: Dict = None):
        self.env = env
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.task_config = task_config
        self.component_aliases = component_aliases
        self.num_components = len(component_aliases)
        self.root = MCTSNode({}, num_total_components=self.num_components)
        self.simulation_cache = {}
        self.best_reward_so_far = -1.0
        self.best_state_so_far = None
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal:
            if node.untried_actions: return node
            if not node.children: return node
            node = max(node.children, key=lambda n: n.ucb1(self.exploration_constant))
        return node
    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions: return node
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        next_component_alias = self.component_aliases[node.next_var_index]
        new_state = node.state.copy()
        new_state[next_component_alias] = action
        new_node = MCTSNode(state=new_state, parent=node, num_total_components=self.num_components)
        node.children.append(new_node)
        return new_node
    def _simulate(self, node: MCTSNode) -> float:
        rollout_state = node.state.copy()
        num_components_set = len(rollout_state)
        for i in range(num_components_set, self.num_components):
            alias = self.component_aliases[i]
            rollout_state[alias] = random.choice(MCTS_ACTION_SPACE)
        state_tuple = tuple(sorted(rollout_state.items()))
        if state_tuple in self.simulation_cache:
            return self.simulation_cache[state_tuple]
        else:
            _, s_full, _ = self.env.step(rollout_state)
            reward = calculate_reward(s_full, self.task_config['focused_f_dict'], self.task_config['reward_weight_dict'])
            self.simulation_cache[state_tuple] = reward
            if reward > self.best_reward_so_far:
                self.best_reward_so_far = reward
                self.best_state_so_far = rollout_state
                logger.info(f"New best reward found: {reward:.4f}")
            return reward
    def run(self):
        logger.info(f"Running MCTS for {self.iterations} iterations...")
        for _ in tqdm(range(self.iterations), desc="MCTS Progress"):
            leaf_node = self._select(self.root)
            if leaf_node.is_terminal:
                reward = self._simulate(leaf_node)
                self._backpropagate(leaf_node, reward)
            else:
                new_node = self._expand(leaf_node)
                reward = self._simulate(new_node)
                self._backpropagate(new_node, reward)
    def get_best_solution(self) -> Tuple[Dict, float]:
        if self.best_state_so_far is None:
            logger.warning("No simulations were run or no valid reward was found. Returning empty solution.")
            return {}, 0.0
        return self.best_state_so_far, self.best_reward_so_far
    def save_tree(self, filepath: str):
        logger.info(f"Saving MCTS tree and cache to {filepath}...")
        data_to_save = {'root': self.root, 'cache': self.simulation_cache, 'exploration_constant': self.exploration_constant, 'best_reward_so_far': self.best_reward_so_far, 'best_state_so_far': self.best_state_so_far}
        with open(filepath, 'wb') as f: pickle.dump(data_to_save, f)
        logger.info("Save complete.")
    @classmethod
    def load_from_file(cls, filepath: str, env: FixedTopologyEnv, component_aliases: List[str], task_config: Dict):
        if not os.path.exists(filepath):
            logger.error(f"Save file not found at {filepath}. Cannot load tree.")
            return None
        logger.info(f"Loading MCTS tree and cache from {filepath}...")
        with open(filepath, 'rb') as f: saved_data = pickle.load(f)
        new_mcts = cls(env, iterations=0, component_aliases=component_aliases, exploration_constant=saved_data['exploration_constant'], task_config=task_config)
        new_mcts.root = saved_data['root']
        new_mcts.simulation_cache = saved_data.get('cache', {})
        new_mcts.best_reward_so_far = saved_data.get('best_reward_so_far', -1.0)
        new_mcts.best_state_so_far = saved_data.get('best_state_so_far', None)
        logger.info(f"Load complete. Loaded {len(new_mcts.simulation_cache)} cached simulations. Best reward so far: {new_mcts.best_reward_so_far:.4f}")
        return new_mcts





# ##################################################################
# ####################      主程序入口      ######################
# ##################################################################
if __name__ == "__main__":
    # 1. 创建环境
    fixed_env = FixedTopologyEnv(config=task_config)
    
    # <<< 关键修改点 >>> 从环境中动态获取需要优化的元件列表
    component_aliases_to_optimize = fixed_env.variable_components
    num_vars = len(component_aliases_to_optimize)
    TREE_SAVE_PATH = f"mcts_tree_checkpoint_{num_vars}_vars_917.pkl"

    # 2. 运行基线仿真 (用于对比)
    logger.info("----------- 1. Running Baseline Simulation -----------")
    fixed_env.reset()
    base_eff, base_s_full, _ = fixed_env.run_simulation()
    base_reward = calculate_reward(base_s_full, task_config['focused_f_dict'], task_config['reward_weight_dict'])
    logger.info(f"Baseline Focused Efficiency: {base_eff}")
    logger.info(f"Baseline Reward Score: {base_reward:.4f}")

    # 3. 初始化MCTS：加载或创建新树
    mcts_iterations = 10 # 您可以根据需要调整迭代次数
    
    # <<< 关键修改点 >>> 将动态列表传入
    mcts_optimizer = MCTS.load_from_file(
        TREE_SAVE_PATH, 
        env=fixed_env, 
        component_aliases=component_aliases_to_optimize, 
        task_config=task_config
    )
    if mcts_optimizer:
        logger.info(f"Successfully loaded tree from {TREE_SAVE_PATH}.")
        mcts_optimizer.iterations = mcts_iterations
        mcts_optimizer.env = fixed_env
        mcts_optimizer.task_config = task_config
    else:
        logger.info(f"No checkpoint found or load failed. Creating a new MCTS tree.")
        # <<< 关键修改点 >>> 将动态列表传入
        mcts_optimizer = MCTS(
            env=fixed_env, 
            iterations=mcts_iterations, 
            component_aliases=component_aliases_to_optimize, 
            task_config=task_config
        )

    # ... (后续代码完全不用变) ...
    # 4. 运行 MCTS 优化
    mcts_optimizer.run()

    # 5. 运行结束后，保存更新后的树
    mcts_optimizer.save_tree(TREE_SAVE_PATH)

    # 6. 获取MCTS找到的最佳方案
    final_components, final_reward_from_mcts = mcts_optimizer.get_best_solution()
    
    if not final_components:
        logger.error("MCTS did not find any valid solution. Exiting.")
        sys.exit()

    logger.info("\n----------- MCTS Optimization Finished -----------")
    logger.info(f"Total simulations in cache: {len(mcts_optimizer.simulation_cache)}")
    logger.info(f"Best reward found during search: {final_reward_from_mcts:.4f}")
    logger.info("Best Component Configuration Found:")
    for alias, value in sorted(final_components.items()):
        logger.info(f"  {alias}: {value}")

    # 7. 使用这个最佳方案进行最终的、独立的验证仿真
    logger.info("\n----------- Running Final Verification Simulation -----------")
    
    fixed_env.reset() 
    final_eff, s_full_final, eff_full_final = fixed_env.step(final_components)
    final_eff, s_full_final, eff_full_final = fixed_env.step(final_components)
    final_reward_verified = calculate_reward(s_full_final, task_config['focused_f_dict'], task_config['reward_weight_dict'])
    
    logger.info(f"Final Focused Efficiency: {final_eff}")
    logger.info(f"Final Verified Reward Score: {final_reward_verified:.4f}")
    logger.info(f"Improvement over baseline: {final_reward_verified - base_reward:.4f}")

    # 8. 绘图展示优化结果
    plt.figure(figsize=(12, 6))
    plt.suptitle("MCTS Optimization Results", fontsize=16)
    ax1 = plt.subplot(1, 2, 1)
    base_s11_db = abopt.value_to_db(base_s_full['FEED_1']['s11'])
    base_freq_ghz = base_s_full['FEED_1']['s11'][:, 0] / 1e9
    ax1.plot(base_freq_ghz, base_s11_db, '--', color='gray', label='Baseline S11')
    final_s11_db = abopt.value_to_db(s_full_final['FEED_1']['s11'])
    final_freq_ghz = s_full_final['FEED_1']['s11'][:, 0] / 1e9
    ax1.plot(final_freq_ghz, final_s11_db, 'r', lw=2, label='Optimized S11')
    ax1.set_xlabel("Freq (GHz)"); ax1.set_ylabel("S11 (dB)"); ax1.set_title("S-Parameter")
    ax1.grid(True); ax1.set_ylim(-30, 0); ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    _, _, base_eff_full = fixed_env.run_simulation()
    ax2.plot(base_eff_full['FEED_1'][:, 0], base_eff_full['FEED_1'][:, 1], '--', color='gray', label='Baseline Efficiency')
    ax2.plot(eff_full_final['FEED_1'][:, 0], eff_full_final['FEED_1'][:, 1], "r", lw=2, label='Optimized Efficiency')
    ax2.set_xlabel("Freq (GHz)"); ax2.set_ylabel("Efficiency (dB)"); ax2.set_title("Total Efficiency")
    ax2.grid(True); ax2.set_ylim(-20, 0); ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


from .hmm_probabilities import HMMProbabilities
from .ti_viterbi import ViterbiAlgorithm, SequenceState
from ..candidate_point import get_candidates
from common.spatial_func import distance
from common.trajectory import STPoint, Trajectory
from ..utils import find_shortest_path
from ..route_constructor import construct_path


"""
包含 hmm-lib 处理新时间步所需的所有内容，包括发射和观察概率。
发射概率：p(z|r)，如果车辆实际在路段 r 上，观察到测量 z 的可能性

每个事件不包含的观测（发射）概率和过渡概率
observation：z 表示车辆的位置
candidates：r 表示车辆可能在的路段
emission_log_probabilities：观测概率
transition_log_probabilities：过渡概率
road_paths：存储当前时间戳从前项到当前过渡概率构建出来的路径
"""
class TimeStep:
    def __init__(self, observation, candidates):
        if observation is None or candidates is None:
            raise Exception('observation and candidates must not be null.')
        self.observation = observation
        self.candidates = candidates
        self.emission_log_probabilities = {}
        self.transition_log_probabilities = {}
        # transition -> dist
        self.road_paths = {}

    # 添加当前时间步候选路段的观测概率
    def add_emission_log_probability(self, candidate, emission_log_probability):
        if candidate in self.emission_log_probabilities:
            raise Exception('Candidate has already been added.')
        self.emission_log_probabilities[candidate] = emission_log_probability

    # 添加两个路段的过渡概率
    def add_transition_log_probability(self, from_position, to_position, transition_log_probability):
        transition = (from_position, to_position)
        if transition in self.transition_log_probabilities:
            raise Exception('Transition has already been added.')
        self.transition_log_probabilities[transition] = transition_log_probability

    # 添加两个路段过度之后的路径，包含保存了之前时间步存储的路径
    def add_road_path(self, from_position, to_position, road_path):
        transition = (from_position, to_position)   # 两个路段之间的转移
        if transition in self.road_paths:
            raise Exception('Transition has already been added.')
        self.road_paths[transition] = road_path

"""
地图匹配器
rn：路网
search_dis=50：搜索半径 计算 route 和 great circle 用到的值
sigma=5.0：观测概率计算需要用到的
beta=2.0：过渡概率计算需要用到的
routing_weight='length'：路段权重用长度表示

"""
class TIHMMMapMatcher():
    def __init__(self, rn, rn_dict, grid_dict, mbr, search_dis, sigma=5.0, beta=2.0, routing_weight='length', debug=False):
        self.measurement_error_sigma = search_dis
        self.transition_probability_beta = beta
        self.guassian_sigma = sigma

        self.debug = debug
        self.rn = rn
        self.rn_dict = rn_dict
        self.grid_dict = grid_dict
        self.mbr = mbr
        self.routing_weight = routing_weight

    # 获取匹配的轨迹 如果没有候选的路径以及过渡概率则开始新的匹配
    def match(self, traj):
        """ Given original traj, return map-matched trajectory"""
        # 计算维特比算法得到的轨迹点序列
        seq = self.compute_viterbi_sequence(traj.pt_list) #最优路径序列 维特比算法
        # 确保匹配后的序列与匹配前的序列有相同的长度
        assert len(traj.pt_list) == len(seq), 'pt_list and seq must have the same size'
        mm_pt_list = []
        # 添加匹配后的信息，候选点字段 每个匹配点会有 state 字段？不为空则是匹配成功，有匹配点
        for ss in seq:
            candi_pt = None
            if ss.state is not None:
                candi_pt = ss.state
            data = {'candi_pt': candi_pt}
            mm_pt_list.append(STPoint(ss.observation.lat, ss.observation.lng, ss.observation.time, data))
        # 构建匹配成功的路径
        mm_traj = Trajectory(traj.oid, traj.tid, mm_pt_list)
        return mm_traj

    # 获取匹配的路径
    def match_to_path(self, traj):
        mm_traj = self.match(traj)
        # rn修改成rn_dict
        path = construct_path(self.rn, mm_traj, self.routing_weight)
        return path

    # 创建时间步
    def create_time_step(self, pt):
        time_step = None
        # 获取候选点集 rn 修改成 rn_dict
        candidates = get_candidates(pt, self.rn_dict,self.grid_dict, self.mbr, self.measurement_error_sigma)
        if candidates is not None:
            time_step = TimeStep(pt, candidates)
        return time_step

    # 通过维特比算法计算最优匹配序列
    def compute_viterbi_sequence(self, pt_list):

        seq = []
        # 概率计算器
        probabilities = HMMProbabilities(self.guassian_sigma, self.transition_probability_beta)
        # 维特比算法计算器
        viterbi = ViterbiAlgorithm(keep_message_history=self.debug)
        # 前一个时间步为空，因为是刚开始匹配
        prev_time_step = None
        idx = 0
        # 匹配点个数
        nb_points = len(pt_list)

        # 匹配计算
        while idx < nb_points:
            # 创建时间步
            time_step = self.create_time_step(pt_list[idx])
            # construct the sequence ended at t-1, and skip current point (no candidate error)
            if time_step is None:
                seq.extend(viterbi.compute_most_likely_sequence())
                seq.append(SequenceState(None, pt_list[idx], None))
                viterbi = ViterbiAlgorithm(keep_message_history=self.debug)
                prev_time_step = None
            else:
                self.compute_emission_probabilities(time_step, probabilities)
                if prev_time_step is None:
                    viterbi.start_with_initial_observation(time_step.observation, time_step.candidates,
                                                           time_step.emission_log_probabilities)
                else:
                    self.compute_transition_probabilities(prev_time_step, time_step, probabilities)
                    viterbi.next_step(time_step.observation, time_step.candidates, time_step.emission_log_probabilities,
                                      time_step.transition_log_probabilities, time_step.road_paths)
                if viterbi.is_broken:
                    # construct the sequence ended at t-1, and start a new matching at t (no transition error)
                    seq.extend(viterbi.compute_most_likely_sequence())
                    viterbi = ViterbiAlgorithm(keep_message_history=self.debug)
                    viterbi.start_with_initial_observation(time_step.observation, time_step.candidates,
                                                           time_step.emission_log_probabilities)
                prev_time_step = time_step
            idx += 1
        if len(seq) < nb_points:
            seq.extend(viterbi.compute_most_likely_sequence())
        return seq

    def compute_emission_probabilities(self, time_step, probabilities):
        for candi_pt in time_step.candidates:
            dist = candi_pt.error
            time_step.add_emission_log_probability(candi_pt, probabilities.emission_log_probability(dist))

    def compute_transition_probabilities(self, prev_time_step, time_step, probabilities):
        linear_dist = distance(prev_time_step.observation, time_step.observation)
        for prev_candi_pt in prev_time_step.candidates:
            for cur_candi_pt in time_step.candidates:

                path_dist, path = find_shortest_path(self.rn, prev_candi_pt, cur_candi_pt, self.routing_weight)
                if path is not None:
                    time_step.add_road_path(prev_candi_pt, cur_candi_pt, path)
                    time_step.add_transition_log_probability(prev_candi_pt, cur_candi_pt,
                                                             probabilities.transition_log_probability(path_dist,
                                                                                                      linear_dist))

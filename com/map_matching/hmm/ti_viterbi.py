
"""

时间非齐次马尔可夫过程的维特比算法的实现，
这意味着状态集和状态转移概率不一定对于所有时间步都是固定的。
对于长观察序列，反向指针通常会在一个
一定数量的时间步长。 例如，在将 GPS 坐标与道路匹配时，最后一个
轨迹中的 GPS 位置通常不再影响第一条道路匹配。
这个实现利用了这个事实，让 Java 垃圾收集器
处理无法访问的反向指针。 如果反向指针在一个
固定数量的时间步长，只有 O(t) 后向指针和转换描述符需要存储在内存中。

"""

class ExtendedState:

    def __init__(self, state, back_pointer, observation, transition_descriptor):
        self.state = state
        self.back_pointer = back_pointer
        self.observation = observation
        self.transition_descriptor = transition_descriptor


class SequenceState:
    def __init__(self, state, observation, transition_descriptor):
        self.state = state
        self.observation = observation
        self.transition_descriptor = transition_descriptor


class ForwardStepResult:
    def __init__(self):

        self.new_message = {}
        self.new_extended_states = {}


class ViterbiAlgorithm:
    def __init__(self, keep_message_history=False):
        self.last_extended_states = None
        self.prev_candidates = []

        self.message = None
        self.is_broken = False
        self.message_history = None
        if keep_message_history:
            self.message_history = []

    def initialize_state_probabilities(self, observation, candidates, initial_log_probabilities):
        if self.message is not None:
            raise Exception('Initial probabilities have already been set.')

        initial_message = {}
        for candidate in candidates:
            if candidate not in initial_log_probabilities:
                raise Exception('No initial probability for {}'.format(candidate))
            log_probability = initial_log_probabilities[candidate]
            initial_message[candidate] = log_probability
        self.is_broken = self.hmm_break(initial_message)
        if self.is_broken:
            return
        self.message = initial_message
        if self.message_history is not None:
            self.message_history.append(self.message)
        self.last_extended_states = {}
        for candidate in candidates:
            self.last_extended_states[candidate] = ExtendedState(candidate, None, observation, None)
        self.prev_candidates = [candidate for candidate in candidates]

    def hmm_break(self, message):

        for log_probability in message.values():
            if log_probability != float('-inf'):
                return False
        return True

    def forward_step(self, observation, prev_candidates, cur_candidates, message, emission_log_probabilities,
                     transition_log_probabilities, transition_descriptors=None):
        result = ForwardStepResult()
        assert len(prev_candidates) != 0

        for cur_state in cur_candidates:
            max_log_probability = float('-inf')
            max_prev_state = None
            for prev_state in prev_candidates:
                log_probability = message[prev_state] + self.transition_log_probability(prev_state, cur_state,
                                                                                        transition_log_probabilities)
                if log_probability > max_log_probability:
                    max_log_probability = log_probability
                    max_prev_state = prev_state

            result.new_message[cur_state] = max_log_probability + emission_log_probabilities[cur_state]

            if max_prev_state is not None:
                transition = (max_prev_state, cur_state)
                if transition_descriptors is not None:
                    transition_descriptor = transition_descriptors[transition]
                else:
                    transition_descriptor = None
                extended_state = ExtendedState(cur_state, self.last_extended_states[max_prev_state], observation,
                                               transition_descriptor)
                result.new_extended_states[cur_state] = extended_state
        return result

    def transition_log_probability(self, prev_state, cur_state, transition_log_probabilities):
        transition = (prev_state, cur_state)
        if transition not in transition_log_probabilities:
            return float('-inf')
        else:
            return transition_log_probabilities[transition]

    def most_likely_state(self):

        assert len(self.message) != 0

        result = None
        max_log_probability = float('-inf')
        for state in self.message:
            if self.message[state] > max_log_probability:
                result = state
                max_log_probability = self.message[state]

        assert result is not None
        return result

    def retrieve_most_likely_sequence(self):

        assert len(self.message) != 0

        last_state = self.most_likely_state()

        result = []
        es = self.last_extended_states[last_state]
        while es is not None:
            ss = SequenceState(es.state, es.observation, es.transition_descriptor)
            result.append(ss)
            es = es.back_pointer
        result.reverse()
        return result

    def start_with_initial_observation(self, observation, candidates, emission_log_probabilities):

        self.initialize_state_probabilities(observation, candidates, emission_log_probabilities)

    def next_step(self, observation, candidates, emission_log_probabilities, transition_log_probabilities, transition_descriptors=None):
        if self.message is None:
            raise Exception('start_with_initial_observation() must be called first.')
        if self.is_broken:
            raise Exception('Method must not be called after an HMM break.')
        forward_step_result = self.forward_step(observation, self.prev_candidates, candidates, self.message,
                                                emission_log_probabilities, transition_log_probabilities, transition_descriptors)
        self.is_broken = self.hmm_break(forward_step_result.new_message)
        if self.is_broken:
            return
        if self.message_history is not None:
            self.message_history.append(forward_step_result.new_message)
        self.message = forward_step_result.new_message
        self.last_extended_states = forward_step_result.new_extended_states
        self.prev_candidates = [candidate for candidate in candidates]

    def compute_most_likely_sequence(self):

        if self.message is None:

            return []
        else:
            return self.retrieve_most_likely_sequence()

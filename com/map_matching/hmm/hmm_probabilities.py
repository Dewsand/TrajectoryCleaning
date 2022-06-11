import math


class HMMProbabilities:
    def __init__(self, sigma, beta):
        self.sigma = sigma
        self.beta = beta

    # 计算观测概率 distance是轨迹点与路段的距离 也就是distance是 ||zt - xt,i||great circle
    def emission_log_probability(self, distance):

        return log_normal_distribution(self.sigma, distance)

    # 计算指数概率分布的过渡概率（状态转移概率）
    def transition_log_probability(self, route_length, linear_distance):

        # transition_metric表示dt，dt = | ||zt - zt+1||great circle - ||xt,i - xt+1,j||route |
        transition_metric = math.fabs(linear_distance - route_length)
        return log_exponential_distribution(self.beta, transition_metric)

# 计算观测概率的公式1 得到log(P)
def log_normal_distribution(sigma, x):
    return math.log(1.0 / (math.sqrt(2.0 * math.pi) * sigma)) + (-0.5 * pow(x / sigma, 2))

# P(dt) = (e^(-dt/beta)) / beta 过渡概率的计算 得到 log(P(dt))
def log_exponential_distribution(beta, x):
    return math.log(1.0 / beta) - (x / beta)

"""
    UCB1 algorithm.
"""

from ..base_pkg.Learner import *
import math

class UCB1_Learner(Learner):

    def __init__(self, n_arms, arm_prices):
        """
        :param n_arms:
        """
        super().__init__(n_arms)
        self.average_rewards =np.zeros(self.n_arms)
        self.arm_prices = arm_prices
        self.delta = np.full((1,n_arms),10000000)[0]

    def pull_arm(self):
        """
        :return: index of the most interesting arm from the demand point of view
        """
        idx = np.argmax(self.average_rewards + self.delta)
        return idx

    def pull_arm_revenue(self):
        """
        :return: index of the most interesting arm from the revenue point of view
        """
        idx = np.argmax((self.average_rewards + self.delta)*self.arm_prices)
        return idx

    def update(self, pulled_arm, bernoulli_reward):
        """
        :param pulled_arm:
        :param bernoulli_reward:
        :return:
        """
        self.t += 1
        real_reward = bernoulli_reward * self.arm_prices[pulled_arm]  # calculate the real reward (isBought*price)

        #na(t-1)
        n_a=len(self.rewards_per_arm[pulled_arm])
        if(n_a==0):
            n_a=0.000000001
        self.update_observations(pulled_arm, real_reward)

        self.average_rewards[pulled_arm] = np.sum(self.average_rewards[pulled_arm]) / self.t
        self.delta[pulled_arm] = math.sqrt(2 * math.log(self.t) / (n_a))

    def get_real_reward(self, pulled_arm, bernoulli_reward):
        """
        :param pulled_arm:
        :param bernoulli_reward:
        :return: the real reward price * bernoulli_real
        """
        real_reward = bernoulli_reward * self.arm_prices[pulled_arm]  # calculate the real reward (isBought*price)
        return real_reward

    def get_mean_reward_from_arm(self, arm):
        probability = self.average_rewards + self.delta
        return probability
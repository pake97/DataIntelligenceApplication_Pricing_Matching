"""
    This class extend the base Environment.
    It is used to iterate over the days of the campaign and to compute the reward obtained by the incoming users.
"""

import numpy as np

from ..base_pkg.Environment import Environment
from ..base_pkg.Parameter import *


class Environment1(Environment):

    def __init__(self, n_days, multi_class_handler, n_arms):
        """
        :param initial_date: when the campaign begins
        :param n_days: number of days of the campaign
        :param users_per_day: number of users per day
        :param multi_class_handler: MultiClassHandler object
        :param n_arms: number of arms of the Thomson Sampling algorithm
        """
        super().__init__(n_days)

        #self.round_per_day = users_per_day #|| da mch
        self.count_rounds_today = 0

        self.mch = multi_class_handler

        self.n_arms = n_arms
        self.arm_prices = self.get_candidate_prices()


    def round(self, pulled_arm_ts,pulled_arm_ucb, user):
        """
            This method performs a round taking the probability from the user's class
        :param pulled_arm: arm to pull
        :param user: User object
        :return: reward
        """
        self.count_rounds_today += 1
        # class of the user
        item1_conv_rate = self.mch.get_class(class_type=user).item1_conv_rate
        item2_conv_rate = self.mch.get_class(class_type=user).item2_conv_rate

        # taking the probability from the conversion curve, associated to the pulled_arm
        probability_ts = item1_conv_rate['probabilities'][self.arm_prices['indices'][pulled_arm_ts]]
        probability_ucb = item1_conv_rate['probabilities'][self.arm_prices['indices'][pulled_arm_ucb]]
        opt = self.mch.get_optimal(class_type=user)
        optimal_revenue = opt['price'] * opt['probability']

        probability_item_2 = item2_conv_rate['probabilities'][price_item_2-min_price_item2]*price_item_2

        if probability_ts < 0:
            probability_ts = 1e-3
        if probability_ucb < 0:
            probability_ucb = 1e-3
        reward_ts = np.random.binomial(1, probability_ts)
        reward_ucb = np.random.binomial(1, probability_ucb)
        reward_item2=np.random.binomial(1, probability_item_2)
        return reward_ts,reward_ucb,reward_item2, optimal_revenue


    def reset_round(self):
        """
            to reset the environment
        :return: None
        """
        self.count_rounds_today = 0
        self.reset()

    def get_candidate_prices(self):
        """
            This method return the candidate prices, one price for each arm.
            The "indices" array contains the positions of the specified prices in the aggregate curve
        :return:
        """
        arm_distance = int(self.mch.aggr_demand_curve_item1['prices'].shape[0] / self.n_arms)
        idx = [int(arm_distance * arm) for arm in range(self.n_arms)]
        prices = self.mch.aggr_demand_curve_item1['prices'][idx]
        return {'indices': idx, 'prices': prices}
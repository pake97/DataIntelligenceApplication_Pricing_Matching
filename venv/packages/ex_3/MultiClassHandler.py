"""
    This class is a container of Class objects.
    It is used to compute the aggregate model and to handle the demand curves of the classes.
"""

import numpy as np
import random
from ..base_pkg.Parameter import *


class MultiClassHandler:

    def __init__(self, *classes_configuration):
        """
        :param classes: number of Class type
        """
        self.classes_configuration = classes_configuration  # list of class objects
        self.optimal_price_for_classes = self.get_optimal_price_for_classes()  # dictionary of opts. The keys are the real name of the classes

        self.aggr_demand_curve_item1 = self.get_aggregate_curve_item1()
        self.aggr_demand_curve_item2 = self.get_aggregate_curve_item2()
        self.aggregate_optimal_price = self.get_optimal_price(self.aggr_demand_curve_item1,self.aggr_demand_curve_item2)

    def get_aggregate_curve_item1(self):
        """
        :return: the aggregate curve
        """
        prices = self.classes_configuration[0].item1_conv_rate['prices']

        stack = [class_conf_.item1_conv_rate['probabilities'] for class_conf_ in self.classes_configuration]

        stack = np.stack(stack, axis=1)
        aggr_probability = np.mean(stack, axis=-1)
        return {'prices': prices, 'probabilities': aggr_probability}

    def get_aggregate_curve_item2(self):
        """
        :return: the aggregate curve
        """
        prices = self.classes_configuration[0].item2_conv_rate['prices']

        stack = [class_conf_.item2_conv_rate['probabilities'] for class_conf_ in self.classes_configuration]

        stack = np.stack(stack, axis=1)
        aggr_probability = np.mean(stack, axis=-1)
        return {'prices': prices, 'probabilities': aggr_probability}

    def get_optimal_price_for_classes(self):
        """
        :return: dictionary containing the: aggregate_optimal, class_1_optimal, class_2_optimal, class_3_optimal
        """
        optimal_prices = {}
        for class_conf_ in self.classes_configuration:
            optimal_prices[class_conf_.type] = self.get_optimal_price(class_conf_.item1_conv_rate,class_conf_.item2_conv_rate)
        return optimal_prices

    def get_conv_rate(self, class_type, arm_price):
        for class_conf_ in self.classes_configuration:
            if class_conf_.type == class_type:
                return class_conf_.item1_conv_rate['probabilities'][self.get_true_index(arm_price)]

    def get_optimal_price(self, conv_rate1,conv_rate2):
        """
            This method computes the max area
        :param conv_rate: dictionary containing the: price, probability
        :return:
        """
        areas = conv_rate1['prices'] * conv_rate1['probabilities']+conv_rate1['probabilities']*conv_rate2['probabilities'][price_item_2-min_price_item2]*price_item_2
        idx = np.argmax(areas)
        return {'price': conv_rate1['prices'][idx],
                'probability': conv_rate1['probabilities'][idx]}

    def get_class(self, class_type):
        for class_conf_ in self.classes_configuration:
            if class_type == class_conf_.type:
                return class_conf_

    def get_optimal(self, class_type):
        return self.optimal_price_for_classes[class_type]

    def get_true_index(self, pull_arm):
        n_arms_pricing=20
        arm_distance = int(self.aggr_demand_curve_item1['prices'].shape[0] / n_arms_pricing)
        return int(arm_distance * pull_arm)

    def get_today_users_income(self):
        today_users_income_per_classes=[c.get_num_users() for c in self.classes_configuration]
        users_income=[]
        for i in range(today_users_income_per_classes[0]):
            users_income.append(1)
        for i in range(today_users_income_per_classes[1]):
            users_income.append(2)
        for i in range(today_users_income_per_classes[2]):
            users_income.append(3)
        for i in range(today_users_income_per_classes[3]):
            users_income.append(4)
        random.shuffle(users_income)
        return users_income
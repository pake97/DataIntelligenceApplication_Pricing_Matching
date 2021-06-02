
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import numpy as np
from scipy.stats import truncnorm

from .Parameter import *




class User_class:

    def __init__(self, class_type):
        """
        :param class_type: number from 1 to 4 corresponding to the class type
        """
        self.type = class_type
        self.dayly_mean_user=dayly_mean_user_per_class[class_type-1]
        self.dayly_variance_user = dayly_variance_user_per_class[class_type-1]
        # here we generate one conversion curve
        self.create_conv_rate()

    def get_num_users(self):
        """

        :return: random number following the truncated gaussian distribution of daily users for that class
        """
        a, b = (0 - self.dayly_mean_user) / self.dayly_variance_user, (np.inf - self.dayly_mean_user) / self.dayly_variance_user
        sample = truncnorm.rvs(a, b,loc=self.dayly_mean_user,scale=self.dayly_variance_user, size=1)
        return int(sample)

    def create_conv_rate(self):
        """
            Creating dictionary to store the abrupt phases
        :return:
        """

        x1 = np.linspace(min_price_item1, max_price_item1, max_price_item1-min_price_item1)

        # the function, which is y =  0.0001e^(-(x/1500)^3)
        y1 = (0.001/(self.type))*np.exp(-1*((x1-160*self.type)/max_price_item1)**4)

        self.item1_conv_rate = {'prices': x1, 'probabilities': y1}

        x2 = np.linspace(min_price_item2, max_price_item2, max_price_item2-min_price_item2)

        # the function, which is y =  0.0001e^(-(x/1500)^3)
        y2 = (0.001/(5-self.type)*2)*np.exp(-1*((x2-160*self.type)/max_price_item2)**4)


        self.item2_conv_rate = {'prices': x2, 'probabilities': y2}



    def plot_conversion_rate(self):
        """
            This function plots the curves of the different abrupt phases
        :return:
        """
        #to do


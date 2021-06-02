"""
    In this script we test the learner on the defined environment.
"""

import copy
from multiprocessing import Pool

import numpy as np
import math
import matplotlib.pyplot as plt

from packages.base_pkg.User_class import User_class
from packages.base_pkg.Parameter import *
from packages.ex_3.Environment1 import Environment1
from packages.ex_3.MultiClassHandler import MultiClassHandler
from packages.ex_3.TS_Learner import TS_Learner
from packages.ex_3.UCB1_Learner import UCB1_Learner


def test_exe3(n_experiments=1,
               demand_chart_path='testing_part3_demandcurves.png',
               demand_chart_title='Part 3 - Demand Curves',
               results_chart_path='testing_part3_regrets.png',
               results_chart_title='Part 3 - Regret'):
    np.random.seed(15)

    # one product to sell
    #product = Product(product_config=product_config)

    # initialization of the four classes
    class_1 = User_class(1)
    class_2 = User_class(2)
    class_3 = User_class(3)
    class_4 = User_class(4)

    mch = MultiClassHandler(class_1, class_2, class_3, class_4)

    n_days=365
    #n_arms_pricing=math.ceil((n_days*math.log(n_days))**(0.25))
    n_arms_pricing=20
    environment = Environment1(n_days=n_days,
                multi_class_handler=mch,
                n_arms=n_arms_pricing)

    plt.title(demand_chart_title)
    for class_ in mch.classes_configuration:
        plt.plot(class_.item1_conv_rate['prices'],
                 class_.item1_conv_rate['probabilities'], label="Class "+str(class_.type), linestyle='--')
    plt.plot(mch.aggr_demand_curve_item1['prices'],
             mch.aggr_demand_curve_item1['probabilities'], label='aggregate demand')

    for opt_class_name, opt in mch.optimal_price_for_classes.items():
        plt.scatter(opt['price'],
                    opt['probability'], marker='o', label=f'opt {opt_class_name}')
    plt.scatter(mch.aggregate_optimal_price['price'],
                mch.aggregate_optimal_price['probability'], marker='o', label='opt aggregate')

    plt.xlabel('Price')
    plt.ylabel('Conversion Rate')
    plt.legend()
    plt.savefig(demand_chart_path)
    plt.show()


    rewards_per_experiment_TS = []  # collect all the rewards achieved from the TS
    optimals_per_experiment = []  # collect all the optimals of the users generated
    rewards_per_experiment_UCB1 = []
    for exp in range(n_experiments):
        print("Inizio esperimento {exp}".format(exp=exp))
        #reset the counter for the einvironment
        environment.reset_round()

        ts_learner = TS_Learner(n_arms=n_arms_pricing, arm_prices=environment.arm_prices['prices'])
        ucb1_learner = UCB1_Learner(n_arms=n_arms_pricing, arm_prices=environment.arm_prices['prices'])
        # ts_learner = SWTS_Learner(n_arms=n_arms, arm_prices=env.arm_prices['prices'], window_size=2000)

        optimal_revenues = np.array([])

        while not environment.simulation_has_finished():
            daily_random_users_per_class = mch.get_today_users_income()
            for u in daily_random_users_per_class:
                pulled_arm_ts = ts_learner.pull_arm_revenue()  # optimize by revenue
                pulled_arm_ucb = ucb1_learner.pull_arm_revenue()
                reward_ts,reward_ucb,reward_item2, opt_revenue = environment.round(pulled_arm_ts,pulled_arm_ucb, u)
                ts_learner.update(pulled_arm_ts, reward_ts,reward_item2)
                ucb1_learner.update(pulled_arm_ucb, reward_ucb,reward_item2)
                optimal_revenues = np.append(optimal_revenues, opt_revenue)

            environment.step()
        rewards_per_experiment_TS.append(ts_learner.collected_rewards)
        optimals_per_experiment.append(optimal_revenues)
        rewards_per_experiment_UCB1.append(ucb1_learner.collected_rewards)
        print("fine esperimento {exp}".format(exp=exp))


    """
    args = [{'environment': copy.deepcopy(environment), 'index': idx, 'keep_daily_price': keep_daily_price} for idx in
            range(n_experiments)]  # create arguments for the experiment

    with Pool(processes=1) as pool:  # multiprocessing.cpu_count()
        results = pool.map(execute_experiment, args, chunksize=1)

    for result in results:
        rewards_per_experiment.append(result['collected_rewards'])
        optimals_per_experiment.append(result['optimal_revenues'])
    """
    fig, axs = plt.subplots(2)
    fig.suptitle(results_chart_title)
    # for opt_class_name, opt in mch.classes_opt.items():
    #     area = opt['price'] * opt['probability']
    #     plt.plot(np.cumsum(np.mean(area - rewards_per_experiment, axis=0)),
    #              label='Regret of the ' + opt_class_name.upper() + ' model')

    # Regret computed UN-knowing the class of the users
    area_aggregate = mch.aggregate_optimal_price['price'] * mch.aggregate_optimal_price['probability']
    max_ylim = 0
    """
    stack = [ex for ex in rewards_per_experiment_TS]
    stack = np.stack(stack, axis=0)
    mean_of_experiments_TS=np.mean(stack, axis=0)
    stack = [ex for ex in rewards_per_experiment_UCB1]
    stack = np.stack(stack, axis=0)
    mean_of_experiments_UCB1=np.mean(stack, axis=0)
    stack = [ex for ex in optimals_per_experiment_TS]
    stack = np.stack(stack, axis=0)
    mean_of_optimals_TS = np.mean(stack, axis=0)
    stack = [ex for ex in optimals_per_experiment_UCB1]
    stack = np.stack(stack, axis=0)
    mean_of_optimals_UCB1 = np.mean(stack, axis=0)

"""
    """
    curve = np.cumsum(np.asarray(area_aggregate) - rewards_per_experiment_TS)

    max_ylim = max(max_ylim, np.max(curve))
    plt.plot(np.linspace(0, n_days, curve.shape[0]), curve, alpha=0.2, c='C0')
    x=np.linspace(0, n_days, curve.shape[0])
    y=np.cumsum(np.mean(area_aggregate - rewards_per_experiment_TS, axis=0))
    plt.plot(x,y,label='Mean Regret of the Aggregate Model with TS', c='C0')
    curve = np.cumsum(np.asarray(area_aggregate) - rewards_per_experiment_UCB1)
    max_ylim = max(max_ylim, np.max(curve))
    plt.plot(np.linspace(0, n_days, curve.shape[0]), curve, alpha=0.2, c='C1')
    plt.plot(np.linspace(0, n_days, curve.shape[0]),
             np.cumsum(np.mean(area_aggregate - rewards_per_experiment_UCB1, axis=0)),
             label='Mean Regret of the Aggregate Model with UCB1', c='C1')
    """
    curves=[]
    for rewards in rewards_per_experiment_TS:
        curve = np.cumsum(np.asarray(area_aggregate) - np.asarray(rewards))
        curves.append(curve)
    y=np.mean(curves, axis=0)
    max_ylim = max(max_ylim, np.max(y))
    axs[0].plot(np.linspace(0, n_days, y.shape[0]), y, alpha=0.2,label='Mean Regret of the Aggregate Model with TS', c='C2')

    curves=[]
    for rewards in rewards_per_experiment_UCB1:
        curve = np.cumsum(np.asarray(area_aggregate) - np.asarray(rewards))
        curves.append(curve)
    y=np.mean(curves, axis=0)
    max_ylim = max(max_ylim, np.max(y))
    axs[0].plot(np.linspace(0, n_days, y.shape[0]), y, alpha=0.2,label='Mean Regret of the Aggregate Model with UCB1', c='C3')





    # Below the regret computed knowing the optimal for each user
    """
    for opt, rewards in zip(optimals_per_experiment_TS, rewards_per_experiment_TS):
        plt.plot(np.linspace(0, n_days, curve.shape[0]), np.cumsum(np.asarray(opt) - np.asarray(rewards)), alpha=0.2,
                 c='C1')
    for opt, rewards in zip(optimals_per_experiment_UCB1, rewards_per_experiment_UCB1):
        plt.plot(np.linspace(0, n_days, curve.shape[0]), np.cumsum(np.asarray(opt) - np.asarray(rewards)), alpha=0.2,
                 c='C1')
    """


    axs[1].plot(np.linspace(0, n_days, curve.shape[0]),
             np.cumsum(np.mean(optimals_per_experiment, axis=0) - np.mean(rewards_per_experiment_TS, axis=0)),
             label='Mean Regret of the True Evaluation TS', c='C0')
    axs[1].plot(np.linspace(0, n_days, curve.shape[0]),
             np.cumsum(np.mean(optimals_per_experiment, axis=0) - np.mean(rewards_per_experiment_UCB1, axis=0)),
             label='Mean Regret of the True Evaluation UCB1', c='C1')

    # plt.yscale('log')
    axs[0].legend()
    axs[1].legend()

    plt.ylim([0, max_ylim])

    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.legend()

    plt.savefig(results_chart_path)
    plt.show()

"""
def execute_experiment(args):
    index = args['index']
    env = args['environment']
    keep_daily_price = args['keep_daily_price']

    _, done = env.reset()

    ts_learner = TS_Learner(n_arms=n_arms_pricing, arm_prices=env.arm_prices['prices'])
    # ts_learner = SWTS_Learner(n_arms=n_arms, arm_prices=env.arm_prices['prices'], window_size=2000)
    optimal_revenues = np.array([])

    new_day = True
    while not done:
        user = User(random=True)

        if keep_daily_price:
            if new_day:
                pulled_arm = ts_learner.pull_arm_revenue()  # optimize by revenue
        else:
            pulled_arm = ts_learner.pull_arm_revenue()  # optimize by revenue

        reward, current_date, new_day, done, opt_revenue = env.round(pulled_arm, user)
        ts_learner.update(pulled_arm, reward)
        optimal_revenues = np.append(optimal_revenues, opt_revenue)

    print(str(index) + ' has ended')

    return {'collected_rewards': ts_learner.collected_rewards, 'optimal_revenues': optimal_revenues}
"""

if __name__ == '__main__':
    test_exe3()



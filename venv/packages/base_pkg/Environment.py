import numpy as np

class Environment:
    def __init__(self, n_days):
        """
        :param n_days: duration of the simulation
        """
        self.current_day = 0
        self.n_days = n_days


    def get_current_day(self):
        """
        :return: the current day of the simulation
        """
        return self.current_day

    def step(self):
        """
            Performing a round
            This method updates the index of the current day and return
            a boolean that is False until we haven't finish the iterations
        :return: (current day, done)
        """
        #print("Day " +str(self.current_day))
        self.current_day += 1

    def simulation_has_finished(self):
        return self.current_day >= self.n_days


    def reset(self):
        """
            Resetting the simulation
        :return: (current day, done)
        """
        self.current_day = 0


"""
if __name__ == '__main__':
    # example
    env = Environment(365)
    env.reset()
    print(curr_date)
    while not done:
        env.step()
        print(curr_day)
"""
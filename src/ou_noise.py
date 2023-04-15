import numpy as np


"""
This  defines a class  OUActionNoise,  represents an Ornstein-Uhlenbeck process, 
a stochastic process to add noise to the actions taken by an agent. 
The process generates a sequence of random numbers 
that are correlated in time and have a mean value of zero. 

"""
class OUActionNoise:

    """
    The OUActionNoise class takes four arguments in its constructor:

    mean: the mean value of the process.
    std_dev: the standard deviation of the process.
    theta: a parameter that controls the rate of mean reversion of the process.
    dt: the time step used to simulate the process.
    x_initial: an optional initial value for the process. If not provided, 
    the process is initialized to zero.

    """
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.mean = mean
        self.std_dev = std_dev
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.reset()


    """
    The __call__ method of the class generates the next value in the process 
    by adding a random increment to the previous value. 
    The increment is calculated using the equation for an Ornstein-Uhlenbeck process:

    dx = theta * (mean - x) * dt + std_dev * sqrt(dt) * dW

    where dW is a random increment drawn from a normal distribution 
    with mean 0 and standard deviation 1. The previous value is then updated 
    to the new value, and the new value is returned.

    """
    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x


    """
    The reset method resets the process to its initial state, 
    either the provided initial value or zero if no initial value was provided.
    """
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

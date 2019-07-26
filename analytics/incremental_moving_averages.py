__author__ = 'julia'


def get_new_mean(x_mean_old, x, n):
    # x_mean_n = x_mean_{n-1} + 1/n *(x - x_mean_{n-1})
    return x_mean_old + 1.0 / n * (x - x_mean_old)


def get_new_var(x_var_old, x_mean_old, x, n):
    # x_var_n = (n-1)/n * x_var_{n-1} + (x - x_mean_{n-1}) * (x-x_mean_n)
    return (1.0 / n) * ((n-1) * x_var_old + (x - x_mean_old) * (x - get_new_mean(x_mean_old, x, n)))


def get_new_exp_mean(x_mean_old, x, gamma):
    # x_mean_n =x_mean_{n-1} + gamma * (x - x_mean_{n-1})
    return x_mean_old + gamma * (x - x_mean_old)


def get_new_exp_var(x_var_old, x_mean_old, x, gamma):
    # x_var_n = (1-gamma) x_var_{n-1} + gamma * (x - x_mean_{n-1}) * (x - x_mean_n)
    return (1-gamma) * x_var_old + gamma * (x - x_mean_old) * (x-get_new_exp_mean(x_mean_old, x, gamma))

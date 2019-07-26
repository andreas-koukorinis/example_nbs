from pymeboot.meboot import meboot
import numpy as np


def get_boostrapped_returns(returns, N, method='meboot'):
    # returns np.array of shape (N, no_returns)
    if len(returns) == 1:
        # impossible to boostrap from 1 element : just returning the same
        return np.ones((N, 1)) * returns[0]
    if method == 'meboot':
        return np.array(meboot(returns, J=N))
    else:
        raise ValueError('Boostrap method {} is not known'.format(method))



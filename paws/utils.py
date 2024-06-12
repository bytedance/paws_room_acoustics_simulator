import numpy as np
import matplotlib.pyplot as plt


def mu_law(x, mu=20):
    output = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    return output


def inv_mu_law(x, mu=20):
    output = np.sign(x) * (1 / mu) * ((1 + mu) ** np.abs(x) - 1)
    return output


def analog_to_digital(x, N=256):
    intervals_num = N // 2 - 1
    delta_y = 1. / intervals_num
    return (np.round(x / delta_y) + N // 2).astype(np.uint8)


def digital_to_analog(x, N=256):
    intervals_num = N // 2 - 1
    delta_y = 1. / intervals_num
    return (x.astype(np.float32) - N // 2) * delta_y


def test():

    mu = 10
    x = np.arange(-1, 1, 0.01)
    x2 = mu_law(x, mu)
    x3 = analog_to_digital(x2)
    y1 = digital_to_analog(x3)
    y2 = inv_mu_law(y1, mu)

    plt.plot(x2)
    plt.savefig("_zz.pdf")


if __name__ == '__main__':

    test()
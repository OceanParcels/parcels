import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from scipy import ndimage
from time import time


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
    n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11
    return np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    RandomState(MT19937(SeedSequence(int(round(time() * 1000)))))  # rs =
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for m_i in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_fractal_noise_temporal2d(shape, tsteps, res, octaves=1, persistence=0.5, max_shift=((-1, 1), (-1, 1))):
    RandomState(MT19937(SeedSequence(int(round(time() * 1000)))))  # rs =
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for m_i in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence

    ishape = (tsteps, )+shape
    result = np.zeros(ishape)
    timage = np.zeros(noise.shape)
    for i in range(0, tsteps):
        result[i, :, :] = noise
        sx = np.random.randint(max_shift[0][0], max_shift[0][1], dtype=np.int32)
        sy = np.random.randint(max_shift[1][0], max_shift[1][1], dtype=np.int32)
        ndimage.shift(noise, (sx, sy), timage, order=3, mode='mirror')
        noise = timage
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    noise = generate_perlin_noise_2d((256, 256), (8, 8))
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()

    np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    plt.figure()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.show()

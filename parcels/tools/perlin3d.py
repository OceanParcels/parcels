import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from scipy import ndimage
from time import time


def generate_perlin_noise_3d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    g000 = gradients[0:-1, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:, :, :, 0]-1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1]-1, grid[:, :, :, 2]), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:, :, :, 0]-1, grid[:, :, :, 1]-1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:, :, :, 0]-1, grid[:, :, :, 1], grid[:, :, :, 2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1]-1, grid[:, :, :, 2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:, :, :, 0]-1, grid[:, :, :, 1]-1, grid[:, :, :, 2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:, :, :, 0]) + t[:, :, :, 0]*n100
    n10 = n010*(1-t[:, :, :, 0]) + t[:, :, :, 0]*n110
    n01 = n001*(1-t[:, :, :, 0]) + t[:, :, :, 0]*n101
    n11 = n011*(1-t[:, :, :, 0]) + t[:, :, :, 0]*n111
    n0 = (1-t[:, :, :, 1])*n00 + t[:, :, :, 1]*n10
    n1 = (1-t[:, :, :, 1])*n01 + t[:, :, :, 1]*n11
    return (1-t[:, :, :, 2])*n0 + t[:, :, :, 2]*n1


def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    RandomState(MT19937(SeedSequence(int(round(time() * 1000)))))
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_fractal_noise_temporal3d(shape, tsteps, res, octaves=1, persistence=0.5, max_shift=(1, 1, 1)):
    RandomState(MT19937(SeedSequence(int(round(time() * 1000)))))
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    ishape = (tsteps, )+shape
    result = np.zeros(ishape)
    timage = np.zeros(noise.shape)
    for i in range(0, tsteps):
        result[i, :, :] = noise
        sx = np.random.randint(-max_shift[0], max_shift[0], dtype=np.int32)
        sy = np.random.randint(-max_shift[1], max_shift[1], dtype=np.int32)
        sz = np.random.randint(-max_shift[2], max_shift[2], dtype=np.int32)
        ndimage.shift(noise, (sx, sy, sz), timage, order=3, mode='mirror')
        noise = timage
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    np.random.seed(0)
    noise = generate_fractal_noise_3d((32, 256, 256), (1, 4, 4), 4)

    fig = plt.figure()
    images = [[plt.imshow(layer, cmap='gray', interpolation='lanczos', animated=True)] for layer in noise]
    animation = animation.ArtistAnimation(fig, images, interval=50, blit=True)
    plt.show()

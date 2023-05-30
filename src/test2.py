from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(projection='3d')

x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z) * np.sqrt(2.0 / 3.0)

q = ax.quiver(x, y, z, u, v, w, length=0.1)

frames = 10

def anim(frame_num):
    global q
    t = frame_num / frames
    u = np.sin(np.pi * x * t) * np.cos(np.pi * y * t) * np.cos(np.pi * z * t)
    v = -np.cos(np.pi * x * t) * np.sin(np.pi * y * t) * np.cos(np.pi * z * t)
    w = np.cos(np.pi * x * t) * np.cos(np.pi * y * t) * np.sin(np.pi * z * t)
    q.remove()
    # q.set_UVC(u, v, w)
    q = ax.quiver(x, y, z, u, v, w, length=0.1)

animatio = animation.FuncAnimation(fig, anim, interval=50)

plt.show()
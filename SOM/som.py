import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# def generate_h(nx, ny, alpha, sigma):
#     h = np.zeros((nx, ny))
#     for i in range(nx):
#         for j in range(ny):
#             h[i][j] = 

#             dist = (c[0]-i)**2 + (c[1]-j)**2
#             m[i][j] += alpha*(x-m[i][j])*np.exp(-dist/(2*sigma**2))
#     return m
#     return h


def generate_ref_vec(nx, ny, d):
    ref = np.random.uniform(-1, 1, size=(nx,ny,d))
    return ref

def generate_input(d, rec=2):
    """
    rec
      1 : 四角形
      2 : 丸
    """
    if rec == 1:
        x = np.random.uniform(-1, 1, size=d)
    elif rec == 2:
        x = []
        r = np.random.uniform(0, 1)
        rad = np.random.uniform(0, 2*np.pi)
        x.append(r*math.sin(rad))
        x.append(r*math.cos(rad))
    return x

def get_winner_idx(nodes, x):
    tmp = x-nodes
    tmp = np.sum(tmp**2, axis=2)
    winner_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
    return winner_idx

def update_ref_vec(nx, ny, m, x, c, t, alpha=0.3, sigma=10):
    sigma = 10-t/300
    for i in range(nx):
        for j in range(ny):
            dist = (c[0]-i)**2 + (c[1]-j)**2
            m[i][j] += alpha*(x-m[i][j])*np.exp(-dist/(2*sigma**2))
    return m

def plot_ref_vec(ref_vec, nx, ny, iter, N):
    fig = []
    x = ref_vec[:,:,0]
    y = ref_vec[:,:,1]
    for i in range(nx-1):
        im = plt.plot([x[i,:], x[i+1,:]], [y[i,:], y[i+1,:]], color="red")
        fig.extend(im)
    for i in range(ny-1):
        im = plt.plot([x[:,i], x[:,i+1]], [y[:,i], y[:,i+1]], color="red")
        fig.extend(im)
    fig.append(plt.text(0.0, 1.1, ("iteration = " + str(iter) + "/" + str(N)), ha="center", va="bottom", fontsize="large"))
    return fig

def save_anime_frame(ims, frame):
    ims.append(frame)
    return ims

def save_anime(fig, ims, nx, ny):
    ani = animation.ArtistAnimation(fig, ims, interval=200)
    ani.save(f"./figs/tr_{nx}_{ny}.gif", writer="imagemagick")

def main(nx=20, ny=20, d=2, N=3000):
    """
    Neural Field は, nx*nyの2次元配列
    1次元を考えたいときは, nx=1 にする
    3次元以上にはできない
    Args:
        nx : Neural Field の行方向のノード数
        ny : Neural Field の列方向のノード数 
        d  : 各ノードが保持する参照ベクトルの次元数
        N  : 学習回数
    """
    np.random.seed(0)
    fig = plt.figure()
    ims = [] # アニメーション生成用
    m = generate_ref_vec(nx, ny, d)
    for i in tqdm(range(N+1)):
       x = generate_input(d)
       c = get_winner_idx(m, x)
       m = update_ref_vec(nx, ny, m, x, c, i)
       if i%(N/100) == 0:
        frame = plot_ref_vec(m, nx, ny, i, N)
        ims = save_anime_frame(ims, frame)
    save_anime(fig, ims, nx, ny)

if __name__ == "__main__":
    main()

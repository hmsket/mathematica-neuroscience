import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def generate_nodes(nx, ny, d):
    nodes = np.random.uniform(-1, 1, size=(nx,ny,d))
    return nodes

def generate_input(d):
    x = np.random.uniform(-1, 1, size=d)
    return x

def get_winner_idx(nodes, x):
    tmp = x-nodes
    tmp = np.sum(tmp**2, axis=2)
    winner_idx = np.unravel_index(np.argmin(tmp), tmp.shape)
    return winner_idx

def update_node(nx, ny, m, x, c, alpha=0.3, sigma=0.8):
    for i in range(nx):
        for j in range(ny):
            dist = (c[0]-i)**2 + (c[1]-j)**2
            m[i][j] += alpha*(x-m[i][j])*np.exp(-dist/(2*sigma**2))
    return m

def plot_nodes(nodes, nx, ny, iter, N):
    fig = []
    x = nodes[:,:,0]
    y = nodes[:,:,1]
    # plt.scatter(x, y)
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

def save_anime(fig, ims):
    ani = animation.ArtistAnimation(fig, ims, interval=200)
    ani.save("anime.gif", writer="imagemagick")

def main(nx=1, ny=10, d=2, N=100000):
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
    m = generate_nodes(nx, ny, d)
    for i in tqdm(range(N+1)):
       x = generate_input(d)
       c = get_winner_idx(m, x)
       m = update_node(nx, ny, m, x, c)
       if i%1000 == 0:
        frame = plot_nodes(m, nx, ny, i, N)
        ims = save_anime_frame(ims, frame)
    save_anime(fig, ims)

if __name__ == "__main__":
    main()

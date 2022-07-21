import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_x_prob():
    p = [0.6, 0.1, 0.1, 0.2]
    return p

def generate_signal(p):
    rng = np.random.default_rng()
    x = rng.choice([[0,0], [0,1], [1,0], [1,1]], p=p)
    return x

def generate_observed_data(x, n, sigma):
    rng = np.random.default_rng()
    y = np.array([])
    for i in range(len(n)):
        rnd = rng.normal(x[i], sigma, size=n[i])
        y = np.append(y, rnd)
    return y

def get_x_prob(p, x1, x2):
    idx = x1*2+x2
    p_tmp = p[idx]
    return p_tmp

def get_output(s, theta):
    if s > theta:
        return 1
    else:
        return 0

def generate_s_god(p, y, n, sigma):
    """
    Args:
        p : 信号xの確率分布
        y : 観測データy
        n : 観測数, 信号x[i]からn[i]個観測する(→y)
        sigma : 標準偏差

    Returns:
        h : yから推測するに,x=[1,1]であると確信する度合い
    """
    x = [0, 1]
    nx = len(x)
    h = 0
    for k in range(nx):
        for l in range(nx):
            x_tilda = [x[k], x[l]]
            p_tmp = get_x_prob(p, x_tilda[0], x_tilda[1])
            count = 0
            idx_sum = 0 # 指数部分(idx)の総和
            for i in range(len(n)):
                for j in range(n[i]):
                    idx_sum += 1-2*y[count]+2*y[count]*x_tilda[i]-x_tilda[i]**2
                    count += 1
            idx_sum /= 2*sigma**2
            h += p_tmp*np.exp(idx_sum)
    p11 = get_x_prob(p, 1, 1)
    h /= p11
    s = 1 / h
    return s

def generate_s_template(p, y, sigma):
    p00 = get_x_prob(p, 0, 0)
    p11 = get_x_prob(p, 1, 1)
    idx_sum = 0
    for i in range(len(y)):
        idx_sum += 1-2*y[i]
    idx_sum /= 2*sigma**2
    h = 1 + p00 * np.exp(idx_sum) / p11
    s = 1 / h
    return s

def generate_s_parts(p, y, n, sigma, theta):
    p00 = get_x_prob(p, 0, 0)
    p01 = get_x_prob(p, 0, 1)
    p10 = get_x_prob(p, 1, 0)
    p11 = get_x_prob(p, 1, 1)
    p0 = p00 + p01
    p1 = p10 + p11
    
    idx_sum = 0
    for i in range(n[0]):
        idx_sum += 1-2*y[i]
    idx_sum /= 2*sigma**2
    h1 = 1 + p0 * np.exp(idx_sum) / p1
    s1 = 1 / h1
    if s1 <= theta:
        # z=0となってほしい
        # z=0にするためには，ここで返す値がthetaを下回る必要がある
        # get_output()を無理矢理使うために
        # 0 <= theta <= 1 であることから
        # 絶対にthetaを下回る，-1を返す
        # もっと良い方法があるだろ！
        return -1
    
    idx_sum = 0
    for i in range(n[0], len(y)):
        idx_sum += 1-2*y[i]
    idx_sum /= 2*sigma**2
    h2 = 1 + p10 * np.exp(idx_sum) / p11
    s2 = 1 / h2
    s = s1 + s2
    return s

def save_ROC(FPRs, CDRs):
    # plt.plot(FPRs, CDRs, marker=".", ls="None")
    plt.title("ROC,$n_i=10$")
    plt.xlabel("FPR")
    plt.ylabel("CDR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(f"ROC_n10.png")
    #plt.show()

def main(n=[10,10], sigma=1.0, N=1000):
    """
    Args:
        n : 観測データyの個数. 信号x[i]からは,n[i]個のデータを観測する. len(x)=len(n)であるのに注意
    """

    plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    FPRs_g, CDRs_g = [], []
    FPRs_t, CDRs_t = [], []
    FPRs_p, CDRs_p = [], []
    p = generate_x_prob()
    
    # 信号と観測データを作成
    X, Y = [], []
    for _ in range(N):
        x = generate_signal(p)
        y = generate_observed_data(x, n, sigma)
        X.append(x)
        Y.append(y)
    
    thetas = np.arange(0.0, 1.01, 0.01)
    for theta in tqdm(thetas):
        n11 = 0 # x=[1,1]であった回数
        n_fp_g, n_cd_g = 0, 0
        n_fp_t, n_cd_t = 0, 0
        n_fp_p, n_cd_p = 0, 0
        for i in range(N):
            sg = generate_s_god(p, Y[i], n, sigma)
            st = generate_s_template(p, Y[i], sigma)
            sp = generate_s_parts(p, Y[i], n, sigma, theta)
            zg = get_output(sg, theta)
            zt = get_output(st, theta)
            zp = get_output(sp, theta)
            if (X[i] == [1,1]).all(): # x=[1,1]であるか否か
                n11 += 1
                if zg == 1: # Yesと判定したかどうか
                    n_cd_g += 1
                if zt == 1: # Yesと判定したかどうか
                    n_cd_t += 1
                if zp == 1: # Yesと判定したかどうか
                    n_cd_p += 1
            else:
                if zg == 1: # Yesと判定したかどうか
                    n_fp_g += 1
                if zt == 1: # Yesと判定したかどうか
                    n_fp_t += 1
                if zp == 1: # Yesと判定したかどうか
                    n_fp_p += 1
        fpr_g = n_fp_g / (N - n11)
        cdr_g = n_cd_g / n11
        fpr_t = n_fp_t / (N - n11)
        cdr_t = n_cd_t / n11
        fpr_p = n_fp_p / (N - n11)
        cdr_p = n_cd_p / n11
        FPRs_g.append(fpr_g)
        CDRs_g.append(cdr_g)
        FPRs_t.append(fpr_t)
        CDRs_t.append(cdr_t)
        FPRs_p.append(fpr_p)
        CDRs_p.append(cdr_p)

    plt.plot(FPRs_g, CDRs_g, label="God")
    plt.plot(FPRs_t, CDRs_t, label="Template")
    plt.plot(FPRs_p, CDRs_p, label="Parts")
    save_ROC(FPRs_g, CDRs_g)

if __name__ == "__main__":
    main()
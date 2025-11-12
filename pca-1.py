# %%
# ...existing code...
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%
# データ生成
# パラメータ
sr = 500                 # サンプリング周波数
t = np.linspace(0, 2, sr*2, endpoint=False)
# 元信号（複合正弦波）
clean = 1.0*np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
# 複数チャンネルをシミュレート（同一信号に独立ノイズを加える）
n_channels = 10
noise_std = 0.8
rng = np.random.default_rng(0)
X = np.vstack([clean + noise_std * rng.normal(size=clean.shape) for _ in range(n_channels)]).T
# X の形: (n_samples, n_channels)


# %%
# 生成データの可視化
fig, axs = plt.subplots(11, 1, sharex=True, figsize=(10, 14))
# 上段: 真の信号
axs[0].plot(t, clean, color='C0', linewidth=1.5)
axs[0].set_title('clean signal')
axs[0].grid(True, axis='x', linestyle='--', alpha=0.4)

# 下段: 各チャンネルの観測
for i in range(n_channels):
    ax = axs[i+1]
    ax.plot(t, X[:, i], color=f'C{(i+1) % 10}', alpha=0.7)
    ax.set_ylabel(f'ch{i}', rotation=0, labelpad=30)
    ax.grid(True, axis='x', linestyle='--', alpha=0.2)

axs[-1].set_xlabel('time [s]')
plt.tight_layout()
plt.show()

# %%
#PCA でノイズ除去
n_components = 2  # ここを変えて試す
pca = PCA(n_components=n_components)
X_mean = X.mean(axis=0)
# sklearn PCA は行がサンプル、列が特徴量（ここではチャンネル）
X_centered = X - X_mean
scores = pca.fit_transform(X_centered)
X_recon = pca.inverse_transform(scores) + X_mean  # 復元されたマルチチャンネル信号

# チャンネルを平均して単一の復元信号を得る（複数観測がある場合）
denoised = X_recon.mean(axis=1)
# 比較用にノイズが入った単一チャンネル（例えば最初のチャンネル）
noisy_ch0 = X[:,0]

# プロット
plt.figure(figsize=(10,6))
plt.plot(t, clean, label='clean signal', linewidth=2)
plt.plot(t, noisy_ch0, label='noisy signal (ch0)', alpha=0.6)
plt.plot(t, denoised, label=f'pca denoised (n_components={n_components})', linewidth=2)
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.title('Denoing with PCA')
plt.tight_layout()
plt.show()

# 説明分散比を表示
print("explained variance ratio:", pca.explained_variance_ratio_.round(3))



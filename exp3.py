#中間層のニューロンユニット数を30に変更

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform

# パラメータ
num_samples = 5000  # サンプル数
learning_rate = 0.35  # 学習係数
mid_unit = 30  # 中間層のニューロンユニット数

y = np.zeros(num_samples)  # システム出力の初期化
u = np.zeros(num_samples)  # システム入力の初期化
errors = np.zeros(num_samples)  # 制御誤差の初期化
evaluation = np.zeros(num_samples)  # 評価関数の初期化
loss_history = []  # 損失関数の履歴

# システムの動作を定義
def system_dynamics(y, u, k):
    y_next = y[k-1] - y[k-2] + 0.35*y[k-3] + u[k-1] + 0.6*u[k-2] + 0.3*u[k-3] - 0.1*y[k-1]**3 + 0.1*y[k-1]*y[k-2]
    return y_next

# 目標値 r(k) を定義
def target_function(k):
    x = np.sin(2 * np.pi * k / 50)
    return 1 if x >= 0 else 0.5

# ニューラルネットワークの設計
def create_nn(input_dim, init_range=0.1):
    # 重みとバイアスを±0.1 の範囲の一様乱数で初期化
    initializer = RandomUniform(minval=-init_range, maxval=init_range)
    model = Sequential()
    model.add(Dense(mid_unit, input_dim=input_dim, activation='tanh',
                    kernel_initializer=initializer, bias_initializer=initializer))  # 中間層
    model.add(Dense(1, activation='linear', kernel_initializer=initializer,
                    bias_initializer=initializer))  # 出力層
    model.compile(optimizer='adam', loss='mse')  # 最適化アルゴリズムと損失関数の設定
    return model

input_dim = 5  # 入力次元(r(k), y(k-1), y(k-2), u(k-1), u(k-2))
nn_model = create_nn(input_dim)  # ニューラルネットワークに入力次元を適応

# 目標値 r(k) の生成
r = np.array([target_function(k) for k in range(num_samples)])

# 制御ループと学習
for k in range(3, num_samples):
    input_data = np.array([r[k], y[k-1], y[k-2], u[k-1], u[k-2]]).reshape(1, -1)  # 入力データの整形
    u[k] = nn_model.predict(input_data, verbose=0)  # システムへの入力予測
    y[k] = system_dynamics(y, u, k)  # システムの動作を計算
    errors[k] = r[k] - y[k]  # 制御誤差の計算
    evaluation[k] = 0.5 * errors[k]**2  # 評価関数の計算

    adjusted_train_y = np.array([r[k] * learning_rate])  # 学習係数を適用
    history = nn_model.fit(input_data, adjusted_train_y, epochs=1, batch_size=1, verbose=0)  # エポック数とバッチサイズの調整
    loss_history.append(history.history['loss'][0])  # 損失関数の値を保存

# 結果をプロット
plt.figure(figsize=(12, 12))

# r(k) と y(k) の時系列プロット
plt.subplot(3, 1, 1)
plt.plot(r[:200], label='r(k)')
plt.plot(y[:200], label='y(k)')
plt.xlabel('sampling number k')
plt.ylabel('output')
plt.legend()
plt.title('k ∈ [0, 200]')

plt.subplot(3, 1, 2)
plt.plot(r[4800:5000], label='r(k)')
plt.plot(y[4800:5000], label='y(k)')
plt.xlabel('sampling number k')
plt.ylabel('output')
plt.legend()
plt.title('k ∈ [4800, 5000]')

# 損失関数のプロット
plt.subplot(3, 1, 3)
plt.plot(loss_history, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss function over iterations')

plt.tight_layout()
plt.show()

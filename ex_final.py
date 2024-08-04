import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import RandomUniform

# 設定値
num_points = 5000  # データポイント数
initial_lr = 0.35  # 初期学習率
hidden_units = 70  # 隠れ層のニューロン数

system_output = np.zeros(num_points)  # 出力の初期化
system_input = np.zeros(num_points)  # 入力の初期化
control_errors = np.zeros(num_points)  # 誤差の初期化
objective_function = np.zeros(num_points)  # 目的関数の初期化
loss_values = []  # 損失値の履歴

# システムの動作方程式
def system_dynamics(system_output, system_input, step):
    y_next = system_output[step-1] - system_output[step-2] + 0.35*system_output[step-3] + system_input[step-1] + 0.6*system_input[step-2] + 0.3*system_input[step-3] - 0.1*system_output[step-1]**3 + 0.1*system_output[step-1]*system_output[step-2]
    return y_next

# 目標関数 r(k)
def reference_signal(step):
    x = np.sin(2 * np.pi * step / 50)
    return 1 if x >= 0 else 0.5

# ニューラルネットワークの作成
def build_nn(input_size, weight_range=0.1):
    # 重みとバイアスの初期化範囲
    initializer = RandomUniform(minval=-weight_range, maxval=weight_range)
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_size, activation='tanh',
                    kernel_initializer=initializer, bias_initializer=initializer))  # 隠れ層
    model.add(Dropout(0.4))  # ドロップアウトレイヤーの追加
    model.add(Dense(1, activation='linear', kernel_initializer=initializer,
                    bias_initializer=initializer))  # 出力層
    model.compile(optimizer='rmsprop', loss='mse')  # オプティマイザーと損失関数の設定
    return model

input_size = 5  # 入力の次元数 (r(k), y(k-1), y(k-2), u(k-1), u(k-2))
neural_net = build_nn(input_size)  # ニューラルネットワークの作成

# 目標関数 r(k) の生成
reference = np.array([reference_signal(step) for step in range(num_points)])

# 制御ループと学習プロセス
for step in range(3, num_points):
    input_vector = np.array([reference[step], system_output[step-1], system_output[step-2], system_input[step-1], system_input[step-2]]).reshape(1, -1)  # 入力ベクトルの生成
    system_input[step] = neural_net.predict(input_vector, verbose=0)  # 次の入力を予測
    system_output[step] = system_dynamics(system_output, system_input, step)  # システムの次状態を計算
    control_errors[step] = reference[step] - system_output[step]  # 誤差の計算
    objective_function[step] = 0.5 * control_errors[step]**2  # 目的関数の計算

    adjusted_output = np.array([reference[step] * initial_lr])  # 学習率の適用
    training_history = neural_net.fit(input_vector, adjusted_output, epochs=1, batch_size=1, verbose=0)  # モデルのトレーニング
    loss_values.append(training_history.history['loss'][0])  # 損失値を保存

# 結果のプロット
plt.figure(figsize=(12, 12))

# 目標値とシステム出力の時系列データ
plt.subplot(3, 1, 1)
plt.plot(reference[:200], label='r(k)')
plt.plot(system_output[:200], label='y(k)')
plt.xlabel('サンプリング数 k')
plt.ylabel('出力')
plt.legend()
plt.title('k ∈ [0, 200]')

plt.subplot(3, 1, 2)
plt.plot(reference[4800:5000], label='r(k)')
plt.plot(system_output[4800:5000], label='y(k)')
plt.xlabel('サンプリング数 k')
plt.ylabel('出力')
plt.legend()
plt.title('k ∈ [4800, 5000]')

# 損失関数の履歴
plt.subplot(3, 1, 3)
plt.plot(loss_values, label='Loss')
plt.xlabel('反復回数')
plt.ylabel('損失')
plt.legend()
plt.title('反復回数に対する損失関数の変化')

plt.tight_layout()
plt.show()

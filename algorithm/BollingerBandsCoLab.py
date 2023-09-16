import os
import pickle

import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


def split_bolling_train_data_co(sequence):
    # 确定80%的点以分割数据集
    split_point = int(len(sequence) * 0.8)

    # 使用切片语法来分割数据集
    train_data = sequence[:split_point]
    test_data = sequence[split_point:]

    return train_data, test_data


def get_positional_encoding(seq_len, feature_count):
    pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0.0, feature_count, 2.0) * -(tf.math.log(10000.0) / feature_count))

    positional_encoding = tf.zeros((seq_len, feature_count))

    positional_encoding = positional_encoding.numpy()
    positional_encoding[:, 0::2] = tf.sin(pos * div_term)
    positional_encoding[:, 1::2] = tf.cos(pos * div_term)

    positional_encoding = tf.convert_to_tensor(positional_encoding)
    positional_encoding = positional_encoding[tf.newaxis, ...]
    return positional_encoding


def build_bolling_time_series_transformer_model_co(feature_count, seq_len, num_layers=6, dff=256, num_heads=8,
                                                   dropout_rate=0.5):
    # 定义模型的输入
    inputs = layers.Input(shape=(None, feature_count))

    # 首先添加一个位置编码层来引入序列中每个点的位置信息
    x = inputs
    positional_encoding = get_positional_encoding(seq_len,
                                                  feature_count)  # Adjust 10000 based on your max sequence length
    x = layers.Add()([x, positional_encoding[:, :tf.shape(x)[1], :]])

    # 添加一个transformer层
    for _ in range(num_layers):
        query_value_attention_seq = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_count)(x, x)
        x = layers.Add()([x, query_value_attention_seq])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)  # 添加dropout层

        # 前馈神经网络
        ffn = layers.Dense(units=dff, activation="relu")(x)
        ffn = layers.BatchNormalization()(ffn)
        ffn = layers.Dense(units=feature_count)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)  # 添加dropout层

    # 用数组切片来获取每个序列的最后一个时间步
    x = layers.Lambda(lambda t: t[:, -1, :])(x)

    # 添加一个或多个Dense层来进行分类
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # 使用softmax激活函数的输出层
    break_output = layers.Dense(2, activation="softmax", name="break_output")(x)
    # 创建模型
    model = models.Model(inputs, [break_output])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics="accuracy")

    return model


# 定义学习率调度函数
def lr_schedule_co(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_bolling_model_co(label_feature_list):
    # 检查GPU是否可用
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f'Found {len(physical_devices)} GPU(s)')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print('No GPU found, using CPU')

    col = ['id', 'close_time']

    sequence_length = len(label_feature_list)  # 获取序列长度
    feature_count = label_feature_list[0]['sequence'].shape[1] - len(col)  # 减去标签列、ID列和close_time列

    train_sequences, test_sequences = split_bolling_train_data_co(label_feature_list)

    del label_feature_list

    def gen(sequences):
        for seq in sequences:
            X = seq['sequence'].drop(columns=col).values
            Y = (seq['label'])
            yield X, {"break_output": Y}

    train_dataset = tf.data.Dataset.from_generator(
        lambda: gen(train_sequences),
        output_signature=(
            tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
            {
                "break_output": tf.TensorSpec(shape=(), dtype=tf.float32),
            }
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: gen(test_sequences),
        output_signature=(
            tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
            {
                "break_output": tf.TensorSpec(shape=(), dtype=tf.float32),
            }
        )
    )

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # 定义回调列表
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),  # 停止训练当验证损失不再改善
            ModelCheckpoint(
                filepath='/content/trading_analysis/static/model/bollinger_break_model.keras',
                monitor='val_loss',
                save_best_only=True
            ),  # 保存验证损失最低的模型
            LearningRateScheduler(lr_schedule_co),
        ]

        model = build_bolling_time_series_transformer_model_co(feature_count, sequence_length)

        history = model.fit(
            train_dataset.batch(1),  # Setting batch size to 1 to handle sequences one by one
            validation_data=test_dataset.batch(1),  # Same for the validation dataset
            epochs=50,
            callbacks=callbacks_list
        )

    loss, break_accuracy = model.evaluate(test_dataset.batch(1))

    print(f"Break output accuracy: {break_accuracy * 100:.2f}%")

    model.save("/content/trading_analysis/static/model/bollinger_break_model.keras")
    return model, history


dir_path = '../static/cache/'
dir_path_ = '/content/trading_analysis/static/cache/'
filepath = "feature_perpusdt_5m_290_0.6_32_0.pkl"
fp = dir_path_ + filepath

se = None
if os.path.exists(fp):
    print(f"Loading existing feature file: {fp}")
    with open(fp, 'rb') as f:
        se = pickle.load(f)

m, h = train_bolling_model_co(se)
print(h)

import os
import random

import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


def split_bolling_train_data_co(df):
    train_size = int(len(df) * 0.9)
    train_data, test_data = df[:train_size], df[train_size:]
    return train_data, test_data


def extract_sequence(df):
    sequences = []

    last_break_point = 0

    for index, row in df.iterrows():
        current_label = row['break_label']
        if current_label in {3, 6}:
            # Determine the minimum start index to get at least a length of 12
            t = df.iloc[last_break_point:index + 1]
            if len(t) < 60:
                i = max(0, index - 59)
                t = df.iloc[i:index + 1]
            sequences.append(t)
            last_break_point = index + 1

    if last_break_point < len(df):
        sequences.append(df.iloc[last_break_point:])

    for i, seq in enumerate(sequences):
        if len(seq) > 370:

            # Find the indices of the mid labels
            mid_labels_indices = seq[(seq['break_label'] == 2) |
                                     (seq['break_label'] == 5)].index

            # Identify the start and end indices for the break
            start_break_idx = mid_labels_indices[0]
            end_break_idx = mid_labels_indices[-1]

            # Find the indices to keep (those with TAKE_PROFIT_LABEL as 1)
            take_profit_indices = seq.loc[start_break_idx:end_break_idx][seq['take_profit_label'] == 1].index

            # Determine the indices to potentially remove (excluding the take profit indices)
            potential_remove_indices = set(mid_labels_indices) - set(take_profit_indices)

            # Remove half of the potential_remove_indices
            indices_to_remove = random.sample(potential_remove_indices, len(potential_remove_indices) // 2)

            # Update the sequence in the list
            sequences[i] = seq.drop(index=indices_to_remove)

    return sequences


def build_bolling_time_series_transformer_model_co(df, num_layers=4, dff=128, num_heads=8, dropout_rate=0.5):
    feature_count = df.shape[1] - 4  # 减去标签列、ID列和close_time列

    # 定义模型的输入
    inputs = layers.Input(shape=(None, feature_count))

    # 首先添加一个位置编码层来引入序列中每个点的位置信息
    x = inputs
    positional_encoding_layer = layers.experimental.preprocessing.CategoryEncoding(
        num_tokens=10000, output_mode="binary")
    positional_encoding = positional_encoding_layer(tf.range(start=0, limit=10000, delta=1))
    x = layers.Add()([x, positional_encoding[:tf.shape(x)[1]]])

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

    # 添加一个时间分布式Dense层来代替全局平均池化层
    x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)

    # 新增的全局平均池化层
    x = layers.GlobalAveragePooling1D()(x)

    # 添加一个或多个Dense层来进行分类
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # 使用softmax激活函数的输出层
    break_output = layers.Dense(7, activation="softmax", name="break_output")(x)
    take_profit_output = layers.Dense(1, activation="sigmoid", name="take_profit_output")(x)

    # 创建模型
    model = models.Model(inputs, [break_output, take_profit_output])

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss={"break_output": "sparse_categorical_crossentropy", "take_profit_output": "binary_crossentropy"},
                  loss_weights={"break_output": 1.0, "take_profit_output": 0.5},
                  optimizer=optimizer,
                  metrics={"break_output": "accuracy", "take_profit_output": "accuracy"})

    return model


# 定义学习率调度函数
def lr_schedule_co(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_bolling_model_co(df):
    # 检查GPU是否可用
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f'Found {len(physical_devices)} GPU(s)')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print('No GPU found, using CPU')

    train_data, test_data = split_bolling_train_data_co(df)

    train_sequences = extract_sequence(train_data)
    test_sequences = extract_sequence(test_data)

    col = ['break_label', 'take_profit_label', 'id', 'close_time']

    feature_count = df.shape[1] - len(col)

    def gen(sequences):
        for seq in sequences:
            X = seq.drop(columns=col).values
            y_break = seq['break_label'].values
            y_take_profit = seq['take_profit_label'].values
            yield X, {"break_output": y_break, "take_profit_output": y_take_profit}

    train_dataset = tf.data.Dataset.from_generator(
        lambda: gen(train_sequences),
        output_signature=(
            tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
            {
                "break_output": tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                "take_profit_output": tf.TensorSpec(shape=(None, ), dtype=tf.int32),
            },
        ),
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: gen(test_sequences),
        output_signature=(
            tf.TensorSpec(shape=(None, feature_count), dtype=tf.float32),
            {
                "break_output": tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                "take_profit_output": tf.TensorSpec(shape=(None, ), dtype=tf.int32),
            },
        ),
    )

    del df

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
                filepath='/content/trading_analysis/static/model/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),  # 保存验证损失最低的模型
            LearningRateScheduler(lr_schedule_co),
        ]

        model = build_bolling_time_series_transformer_model_co(train_data)

        history = model.fit(
            train_dataset.batch(1),  # Setting batch size to 1 to handle sequences one by one
            validation_data=test_dataset.batch(1),  # Same for the validation dataset
            epochs=50,
            callbacks=callbacks_list
        )

    loss, break_loss, take_profit_loss, break_accuracy, take_profit_accuracy = model.evaluate(test_dataset.batch(1))

    print(f"Break output accuracy: {break_accuracy * 100:.2f}%")
    print(f"Take profit output accuracy: {take_profit_accuracy * 100:.2f}%")

    return model, history


filepath = '/content/trading_analysis/static/cache/feature_perpusdt_5m_290_0.6_32_0.csv'
#  filepath_ = '../static/cache/feature_perpusdt_5m_290_0.6_32_0.csv'

df = None
if os.path.exists(filepath):
    print(f"Loading existing feature file: {filepath}")
    df = pd.read_csv(filepath)
m, h = train_bolling_model_co(df)

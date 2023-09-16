import random

import tensorflow as tf
from keras import layers, models
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from algorithm.FeatureEngineering import create_bollinger_bands_feature
from algorithm.common import *
from foundation.Logger import logger


def split_bolling_train_data(df):
    train_size = int(len(df) * 0.9)
    train_data, test_data = df[:train_size], df[train_size:]
    return train_data, test_data


def build_bolling_time_series_transformer_model(feature_label_map, num_layers=8, dff=256, num_heads=12, dropout_rate=0.5):
    sequence_length = len(feature_label_map)  # 获取序列长度
    feature_count = len(feature_label_map[0]["sequence"][1] - 4)  # 减去标签列、ID列和close_time列

    # 定义模型的输入
    inputs = layers.Input(shape=(None, feature_count))

    # 首先添加一个位置编码层来引入序列中每个点的位置信息
    positional_encoding = layers.Embedding(input_dim=sequence_length, output_dim=feature_count)(
        tf.range(start=0, limit=sequence_length, delta=1))
    x = inputs + positional_encoding

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
    optimizer = tf.keras.optimizers.legacy.Adam()

    model.compile(loss={"break_output": "sparse_categorical_crossentropy", "take_profit_output": "binary_crossentropy"},
                  loss_weights={"break_output": 1.0, "take_profit_output": 0.5},
                  optimizer=optimizer,
                  metrics={"break_output": "accuracy", "take_profit_output": "accuracy"})

    return model


# 定义学习率调度函数
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_bolling_model(df):
    train_data, test_data = split_bolling_train_data(df)

    model = build_bolling_time_series_transformer_model(train_data)

    col = [BREAK_LABEL, TAKE_PROFIT_LABEL, 'id', 'close_time']

    X_train = train_data.drop(columns=col)
    y_train_break = train_data[BREAK_LABEL]
    y_train_take_profit = train_data[TAKE_PROFIT_LABEL]

    X_test = test_data.drop(columns=col)
    y_test_break = test_data[BREAK_LABEL]
    y_test_take_profit = test_data[TAKE_PROFIT_LABEL]

    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    del df

    # 定义回调列表
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),  # 停止训练当验证损失不再改善
        ModelCheckpoint(
            filepath='../static/model/best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),  # 保存验证损失最低的模型
        LearningRateScheduler(lr_schedule),
    ]

    history = model.fit(
        X_train,
        {"break_output": y_train_break, "take_profit_output": y_train_take_profit},
        validation_data=(X_test, {"break_output": y_test_break, "take_profit_output": y_test_take_profit}),
        epochs=50,  # 你可以根据需要更改epoch的数量
        batch_size=36,  # 你可以更改batch size
        callbacks=callbacks_list  # 为模型训练添加回调列表
    )

    loss, break_loss, take_profit_loss, break_accuracy, take_profit_accuracy = model.evaluate(
        X_test, {"break_output": y_test_break, "take_profit_output": y_test_take_profit})

    logger.info(f"Break output accuracy: {break_accuracy * 100:.2f}%")
    logger.info(f"Take profit output accuracy: {take_profit_accuracy * 100:.2f}%")

    return model, history


d = create_bollinger_bands_feature('perpusdt', '5m', 290, 0.6, 32, 0)
se = extract_sequence(d)
# for i, seq in enumerate(sequences):
#     print(f"Sequence {i} BREAK_LABEL sequence:")
#     print(seq[BREAK_LABEL].to_list())

max_length = max(len(seq) for seq in se)
print(f"The maximum sequence length is: {max_length}")

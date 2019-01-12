# https://www.tensorflow.org/guide/checkpoints
import tensorflow as tf

# save checkpoints and other files to model_dir
my_feature_columns = []
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris')

# save checkpoints config
# 默认参数下
# 每 10 分钟（600 秒）写入一个检查点
# 在 train 方法开始（第一次迭代）和完成（最后一次迭代）时写入一个检查点
# 只在目录中保留 5 个最近写入的检查点
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max=10,       # Retain the 10 most recent checkpoints.
)
classifier2 = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris',
    config=my_checkpointing_config)

# 后续调用train、evaluate、predict，从checkpoint恢复模型

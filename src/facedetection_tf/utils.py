import tensorflow as tf
import keras as K
from src.facedetection_tf.logger import FD_Logger
import json

logger = FD_Logger().logger()

def limit_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus)==0:
        logger.info("No GPU's found to limit memory growth")
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth set")

def load_image(x):
    byte_img=tf.io.read_file(x)
    img=tf.io.decode_jpeg(byte_img)
    return img


def load_labels(label_file_path):
    with open(label_file_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)
    return [label["class_label"]], list(label["bbox"].values())


def get_optimizer(batches_per_epoch:int, lr:float):
    """
    Uses Adam optimizer

    >>> batches_per_epoch = len(train_ds)
    >>> lr_decay = (1.0 / 0.75 - 1) / batches_per_epoch

    """
    lr_decay = (1.0 / 0.75 - 1) / batches_per_epoch
    optimizer = K.optimizers.Adam(learning_rate=lr, decay=lr_decay)
    return optimizer


def get_localization_loss(y_true, y_hat):
    delta_coords = tf.reduce_sum(tf.square(y_true[:,:2] - y_hat[:,:2]))

    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]

    h_pred = y_hat[:,3] - y_hat[:,1]
    w_pred = y_hat[:,2] - y_hat[:,0]

    delta_size = tf.reduce_sum(tf.square(h_true-h_pred) + tf.square(w_true+w_pred))

    return delta_coords + delta_size


def get_categorical_loss(**kwargs):
    return K.losses.BinaryCrossentropy(**kwargs)

def add_from_kwargs(cls):
    def from_kwargs(cls, *args, **kwargs):
        try:
            initializer = cls.__initializer
        except AttributeError:
            # Store the original init on the class in a different place
            cls.__initializer = initializer = cls.__init__
            # replace init with something harmless
            cls.__init__ = lambda *a, **k: None

        # code from adapted from Arne
        added_args = {}
        for name in list(kwargs.keys()):
            if name not in cls.__annotations__:
                added_args[name] = kwargs.pop(name)

        ret = object.__new__(cls)
        initializer(ret, **kwargs)
        # ... and add the new ones by hand
        for new_name, new_val in added_args.items():
            setattr(ret, new_name, new_val)
        return ret
    cls.from_kwargs = classmethod(from_kwargs)
    return cls


if __name__=="__main__":
    __all__=["limit_memory_growth","load_image","load_labels","get_optimizer","get_localization_loss","get_categorical_loss"]
from model_loader import *
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

#tf.config.run_functions_eagerly(True)

path = '/share/Disk_2/Projects/trader/data/'
dataset = 'binance-1h'

p = path + 'Binance_BTCUSDT_1h.csv'


# unet = UNet(img_size=, num_classes=2, weights=weights_path, lr=3e-4)
# pspnet = PSPNet(img_size=(480, 480), num_classes=2, weights=weights_path, lr=3e-4)
# fcn16s = FCN16s(img_size=(512, 512), num_classes=2, weights=weights_path, lr=3e-4)
# segnet = SegNet(img_size=(512, 512), num_classes=2, weights=weights_path, lr=3e-4)

RUN = 0


models_data = {
        'LSTMNet': {'model': LSTMNet, 'input_shape': (80, 1), 'weights_path': None, 'lr': 1e-3}, 
             
}

for model_name in models_data:
        model_data = models_data[model_name]
        data_loader = BinanceDataLoader(p, window_size=model_data['input_shape'][0], predict_size=1)
        train, val = data_loader.get_data()
        train_size, val_size = data_loader.get_train_size(), data_loader.get_val_size()

        

        keras.backend.clear_session()

        model_class = model_data['model'](model_data['input_shape'], output_size=1, weights=model_data['weights_path'], lr=model_data['lr'])
        model = model_class.get_model()



        BATCH_SIZE = 32

        STEPS_PER_EPOCH = train_size // BATCH_SIZE 
        VALIDATION_STEPS = val_size // BATCH_SIZE 

        t = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")

        m = model.fit(
                train,
                validation_data=val,
                batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_batch_size=BATCH_SIZE,
                validation_steps=VALIDATION_STEPS,
                callbacks=model_class.get_callbacks(
                        num_epochs=100,
                        save_weights='/share/Disk_2/Projects/trader/weights/{}/{}-{}'.format(dataset, model_name, t),
                        dataset_name=dataset,
                        log_dir='/share/Disk_2/Tensorboard/trader/{}/{}-{}'.format(dataset, model_name, t)
                        ),
                epochs=600)


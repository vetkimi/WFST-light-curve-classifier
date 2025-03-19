#VAE model training
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

#loading your data in correct format
#input: [mask_data, mask_labeled_data], target: [data, labeled_label]

#loadding data
labeled_data_file = np.load("/home/yltang/data/WFST_lc_classification/data/train_data.npz")
labeled_data = labeled_data_file["data"]
labeled_label_file = np.load("/home/yltang/data/WFST_lc_classification/data/train_label.npz")
labeled_label = labeled_label_file["data"]
unlabeled_data_file = np.load("/home/yltang/data/WFST_lc_classification/data/test_data.npz")
unlabeled_data = unlabeled_data_file["data"]

print(labeled_data.shape)
data = np.concatenate([labeled_data, unlabeled_data], axis=0)
data_clone = labeled_data
label_clone = labeled_label

for i in range(5):
    labeled_data = np.concatenate([labeled_data, data_clone], axis=0)
    labeled_label = np.concatenate([labeled_label, label_clone], axis=0)
labeled_data = labeled_data[:data.shape[0]]
labeled_label = labeled_label[:data.shape[0]]

def data_normal(data, max_flux):
    #max_flux = np.max(data[:, :, :3])
    for i in range(data.shape[0]):
        max_flux_ = np.max(data[i, :, :3])
        for j in range(3):
            data[i, :, j] = (data[i, :, j]/(np.abs(max_flux_)+1)) * (np.log(np.abs(max_flux_+10)+10)/np.log(max_flux))
            data[i, :, j+3] = (data[i, :, j+3]/(np.abs(max_flux_)+1)) *  (np.log(np.abs(max_flux_+10)+10)/np.log(max_flux))
    return data

mask_time = 6
def data_mask(data):
    data[:, mask_time:, :]=0
    return data

max_flux = np.max(labeled_data[:, :, :3])
mask_data = data_mask(data=data.copy())
mask_labeled_data = data_mask(data=labeled_data.copy())

mask_data = data_normal(data=mask_data, max_flux=max_flux)
mask_labeled_data = data_normal(data=mask_labeled_data, max_flux=max_flux)
data = data_normal(data=data, max_flux=max_flux)

#input shape
num_time_steps = data.shape[1]
num_features = data.shape[2]
num_time_steps, num_features
latent_dim = 2

# model
encoder_inputs = keras.Input(shape=(90, num_features))
x_1 = layers.Masking(mask_value=0.0)(encoder_inputs)
x_1 = layers.GRU(100,return_sequences=True)(x_1)
x_1 = keras.layers.LayerNormalization()(x_1)
x_1 = layers.GRU(100)(x_1)
x_1 = keras.layers.LayerNormalization()(x_1)
x_1 = keras.layers.Dense(64, activation="relu")(x_1)
x_1 = keras.layers.Dense(32, activation="relu")(x_1)
#concate the convolutional branch and the recurrent branch
#concated_information = layers.concatenate([x_1,x_2])
z_mean = layers.Dense(latent_dim, name="z_mean")(x_1)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_1)
classification_probability = keras.layers.Dense(9, activation="softmax",name="classification_probability")(x_1)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, classification_probability], name="encoder")

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(90*100, activation="relu")(latent_inputs)
x = layers.Reshape((90, 100))(x)
x = keras.layers.LayerNormalization()(x)
x = layers.GRU(100,return_sequences=True)(x)
x = keras.layers.LayerNormalization()(x)
x = layers.GRU(100,return_sequences=True)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
decoder_outputs = layers.Dense(num_features, activation="relu")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

class Sampler(layers.Layer):
    def call(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon * 0.1
    
classification_weight = 10
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.classification_tracker = keras.metrics.Mean(name="classification_loss")
    @property
    def metrics(self):
        return [self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
                self.classification_tracker
               ]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, a= self.encoder(data[0][0])
            b,c,classification_probability = self.encoder(data[0][1])
            z = self.sampler(z_mean, z_log_var)
            reconstruction = decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mse(data[1][0], reconstruction)
                )
            )
            """
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(data[1][0],reconstruction)
            )*reconstruction_weight
            """
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            classification_loss = keras.losses.categorical_crossentropy(data[1][1], classification_probability)*classification_weight
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss) + tf.reduce_mean(classification_loss)

            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_tracker.update_state(classification_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "classification_loss": self.classification_tracker.result()
        }


#training
epochs = 10000
batch_size = 1024
vae = VAE(encoder, decoder)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model_1.keras", save_best_only=True, monitor="total_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="total_loss", factor=0.5, patience=100, min_lr=0.00001
    ),
    keras.callbacks.EarlyStopping(monitor="total_loss", patience=200, verbose=1, mode='min'),
]
vae.compile(
    optimizer="rmsprop",
    run_eagerly=True
)

history = vae.fit(
    [mask_data, mask_labeled_data],
    [data, labeled_label],
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    #validation_data=validation_data,
    verbose=1,
)

#saving model
vae.encoder.save(f"LSTM_encoder.keras")
vae.decoder.save(f"LSTM_decoder.keras")

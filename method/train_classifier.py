import sys
import numpy as np

#loading data
labeled_data_file = np.load("/home/yltang/data/work_1/data_3/train_labeled_data.npz")
labeled_data = labeled_data_file["data"]
labeled_label_file = np.load("/home/yltang/data/work_1/data_3/train_labeled_label.npz")
labeled_label = labeled_label_file["data"]
unlabeled_data_file = np.load("/home/yltang/data/work_1/data_3/train_unlabeled_data.npz")
unlabeled_data = unlabeled_data_file["data"]
print(labeled_data.shape)
data = np.concatenate([labeled_data, unlabeled_data], axis=0)
data_clone = labeled_data
label_clone = labeled_label
for i in range(6):
    labeled_data = np.concatenate([labeled_data, data_clone], axis=0)
    labeled_label = np.concatenate([labeled_label, label_clone], axis=0)
labeled_data = labeled_data[:data.shape[0]]
labeled_label = labeled_label[:data.shape[0]]

#data normalization
def data_normal(data, max_flux):
    #max_flux = np.max(data[:, :, :3])
    for i in range(data.shape[0]): 
        max_flux_ = np.max(data[i, :, :3])
        for j in range(3):
            data[i, :, j] = (data[i, :, j]/(np.abs(max_flux_)+1)) * (np.log(np.abs(max_flux_+10)+10)/np.log(max_flux))
            data[i, :, j+3] = (data[i, :, j+3]/(np.abs(max_flux_)+1)) *  (np.log(np.abs(max_flux_+10)+10)/np.log(max_flux))
    return data
# data mask
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

# build the model
import tensorflow as tf
from tensorflow import keras
from keras import layers

#input shape
num_time_steps = data.shape[1]
num_features = data.shape[2]
num_time_steps, num_features
latent_dim = 2

#loadding trained VAE model
from keras.models import load_model
vae_decoder = load_model(f"/home/yltang/data/work_1/results_1/LSTM_decoder.keras")
vae_encoder = load_model(f"/home/yltang/data/work_1/results_1/LSTM_encoder.keras")

#Get all layers of vae_encoder except the last one
partial_vae_encoder = Model(
    inputs=vae_encoder.input,
    outputs=vae_encoder.layers[-6].output
)

#Freeze all layers of partial_vae_encoder
for layer in partial_vae_encoder.layers:
    layer.trainable = False

#Get the output of the new model
new_output = partial_vae_encoder.output

#Add new fully connected layers and give each layer a unique name
x = layers.Dense(64, activation='relu', name='dense_1')(new_output)
x = layers.Dense(32, activation='relu', name='dense_2')(x)
predictions = layers.Dense(9, activation='softmax', name='dense_output')(x)

#Building a new model
vae_model = Model(inputs=partial_vae_encoder.input, outputs=predictions)

#Print the structure of the new model to confirm that everything works
vae_model.summary()


"""
# benchmark model
encoder_inputs = keras.Input(shape=(90, num_features))
#recurrent branch0
x_1 = layers.Masking(mask_value=0.0)(encoder_inputs)
x_1 = layers.GRU(100,return_sequences=True)(x_1)
x_1 = keras.layers.LayerNormalization()(x_1)
x_1 = layers.GRU(100)(x_1)
x_1 = keras.layers.LayerNormalization()(x_1)
x_1 = keras.layers.Dense(64, activation="relu")(x_1)
x_1 = keras.layers.Dense(32, activation="relu")(x_1)
classification_probability = keras.layers.Dense(9, activation="softmax",name="classification_probability")(x_1)
rnn = keras.Model(encoder_inputs, classification_probability, name="rnn")

rnn.summary()
"""

#training
epochs = 5000
batch_size = 512

callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"vae_model.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=1),
]
vae_model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["acc"],
)
history = vae_model.fit(
    mask_labeled_data,
    labeled_label,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

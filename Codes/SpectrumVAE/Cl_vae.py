"""
Variational Auto-encoder for dimensionality reduction. 

Model is applied on Cosmic Microwave Background - angular power spectrum data 
Reduced latent variables are GP fitted with Cosmological parameters used to generate them.

"""

from tensorflow.examples.tutorials.mnist import input_data # Works on MNIST data
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import losses


# from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


original_dim = 2549 # For MNIST ~ 784
intermediate_dim1 = 1024 #
intermediate_dim = 512 #
latent_dim = 10

totalFiles = 256 
TestFiles = 32


batch_size = 8
num_epochs = 100
epsilon_mean = 1.0 
epsilon_std = 1.0 
learning_rate = 1e-7
decay_rate = 0.0



# Q(z|X) -- encoder
inputs = Input(shape=(original_dim,))
h_q1 = Dense(intermediate_dim1, activation='relu')(inputs) # ADDED intermediate layer
h_q = Dense(intermediate_dim, activation='relu')(h_q1)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

# ----------------------------------------------------------------------------

def sample_z(args):
    mu, log_sigma = args
    ###eps = K.random_normal(shape=(m, n_z), mean=0., std=1.)
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=epsilon_mean, stddev=epsilon_std)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# ----------------------------------------------------------------------------

# P(X|z) -- decoder
decoder_hidden = Dense(latent_dim, activation='relu')
decoder_hidden1 = Dense(intermediate_dim, activation='relu') # ADDED intermediate layer
decoder_hidden2 = Dense(intermediate_dim1, activation='relu') # ADDED intermediate layer
decoder_out = Dense(original_dim, activation='sigmoid')

h_p1 = decoder_hidden(z)
h_p2 = decoder_hidden1(h_p1) # ADDED intermediate layer
h_p3 = decoder_hidden2(h_p2) # ADDED intermediate layer
outputs = decoder_out(h_p3)

# ----------------------------------------------------------------------------


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_hidden(decoder_input)
_h1_decoded = decoder_hidden1(_h_decoded)    ## ADDED layer_1
_h0_decoded = decoder_hidden2(_h1_decoded)    ## ADDED --- should replicate decoder arch
_x_decoded_mean = decoder_out(_h0_decoded)
decoder = Model(decoder_input, _x_decoded_mean)


# -------------------------------------------------------------
#CUSTOM LOSS

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # recon = K.categorical_crossentropy(y_pred, y_true)

    # recon = losses.mean_squared_error(y_pred, y_true)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

#----------------------------MNIST load ------------------------------
# MNIST or any flattened arrays work here

# from keras.datasets import mnist 
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# X_train = x_train.astype('float32') / 255.
# ## X_test = x_test.astype('float32') / 255.
# X_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# ## X_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
# X_train = mnist.train.images
# X_train = X_train.astype('float32') / 255.
#
# X_test = mnist.test.images
# X_test = X_test.astype('float32') / 255.
# Y_test = mnist.test.labels
# -------------------------------------------------------------


# ----------------------------- i/o ------------------------------------------

import Cl_load

Dir0 = '../../AllTrainTestSets/'
train_path = Dir0+'Cl_data/Data/LatinCl_'+str(totalFiles)+'.npy'
train_target_path = Dir0+ 'Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy'
test_path = Dir0+'Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy'
test_target_path = Dir0+ 'Cl_data/Data/LatinPara5_'+str(TestFiles)+'.npy'


camb_in = Cl_load.cmb_profile(train_path = train_path,  train_target_path = train_target_path , test_path = test_path, test_target_path = test_target_path, num_para=5)


(x_train, y_train), (x_test, y_test) = camb_in.load_data()

x_train = x_train[:,2:]
x_test = x_test[:,2:]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')


normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.
np.save(Dir0+'Cl_data/Data/normfactor_'+str(totalFiles)+'.npy', normFactor)


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# Adding noise factor for denoising VAE. 
noise_factor = 0.00

x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# plt.plot(x_train_noisy.T)
# ------------------------------------------------------------------------------

#TRAIN    
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate)
vae.compile(optimizer='adam', loss=vae_loss)
vae.fit(x_train_noisy, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2,
        validation_data=(x_test_noisy, x_test))


# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_decoded = decoder.predict(x_train_encoded)

np.save(Dir0+'Cl_data/Data/encoded_xtrain_'+str(totalFiles)+'.npy', x_train_encoded)

# ----------------------------------------------------------------------------

SaveModel = False
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']

    training_hist = np.vstack([epochs, train_loss, val_loss])

    # fileOut = 'Stack_opti' + str(opti_id) + '_loss' + str(loss_id) + '_lr' + str(learning_rate) + '_decay' + str(decay_rate) + '_batch' + str(batch_size) + '_epoch' + str(num_epoch)

    fileOut = 'DenoiseModel_'+str(totalFiles)
    vae.save(Dir0+'Cl_data/Model/fullAE_' + fileOut + '.hdf5')
    encoder.save(Dir0+'Cl_data/Model/Encoder_' + fileOut + '.hdf5')
    decoder.save(Dir0+'Cl_data/Model/Decoder_' + fileOut + '.hdf5')
    np.save(Dir0+'Cl_data/Model/TrainingHistory_'+fileOut+'.npy', training_hist)


# --------------------- Plotting routines -------------------------------------

ls = np.load(Dir0+'Cl_data/Data/ls_'+str(totalFiles)+'.npy')[2:]
# display a 2D plot of the digit classes in the latent space
PlotScatter = False
if PlotScatter:
    plt.figure(figsize=(6, 6))
    x_train_encoded = encoder.predict(x_train)
    plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train[:, 0], cmap='spring')
    plt.colorbar()

    x_test_encoded = encoder.predict(x_test)
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:, 0], cmap='copper')
    plt.colorbar()
    plt.show()


# ----------------------------------------------------------------------------

PlotSample = False
if PlotSample:
    for i in range(3,10):
        plt.figure(91, figsize=(8,6))
        plt.plot(ls, x_decoded[i], 'r--', alpha = 0.8)
        plt.plot(ls, x_train[i], 'b--', alpha = 0.8)
        # plt.xscale('log')
        # plt.yscale('log')
        plt.title('reconstructed - red')

    plt.show()


# ----------------------------------------------------------------------------
plotLoss = False
if plotLoss:
    import matplotlib.pylab as plt

    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    val_loss = vae.history.history['val_loss']


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    plt.tight_layout()
    # plt.savefig('../Cl_data/Plots/Training_loss.png')

plt.show()


# ----------------------------------------------------------------------------
PlotModel = False
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = Dir0+'Cl_data/Plots/ArchitectureFullAE.png'
    plot_model(vae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = Dir0+'Cl_data/Plots/ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = Dir0+'Cl_data/Plots/ArchitectureDecoder.png'
    plot_model(decoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

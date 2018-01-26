"""
GP fit for W matrix - with only 2 eigenvalues

Uses George - package by Dan Foreman McKay - better integration with his MCMC package.
pip install george  - http://dan.iel.fm/george/current/user/quickstart/

Higdon et al 2008, 2012

"""
print(__doc__)


import numpy as np



from matplotlib import pyplot as plt


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


totalFiles = 256
TestFiles = 32

latent_dim = 10


import george
from george.kernels import Matern32Kernel, ConstantKernel, WhiteKernel

kernel = Matern32Kernel(0.5, ndim=5)


# ----------------------------- i/o ------------------------------------------

import Cl_load

train_path = '../Cl_data/Data/LatinCl_'+str(totalFiles)+'.npy'
train_target_path =  '../Cl_data/Data/LatinPara5_'+str(totalFiles)+'.npy'
test_path = '../Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy'
test_target_path =  '../Cl_data/Data/LatinPara5_'+str(TestFiles)+'.npy'


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
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# ------------------------------------------------------------------------------


X1 = y_train[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = y_train[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = y_train[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = y_train[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

X5 = y_train[:, 4][:, np.newaxis]
X5a = rescale01(np.min(X5), np.max(X5), X5)


XY = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]


# ------------------------------------------------------------------------------
# This part will go inside likelihood 
# # ------------------------------------------------------------------------------
y = np.load('../Cl_data/Data/encoded_xtrain_'+str(totalFiles)+'.npy').T
# # ------------------------------------------------------------------------------
# Decoder acts here

from keras.models import load_model

fileOut = 'DenoiseModel_'+str(totalFiles)
# vae = load_model('../Pk_data/fullAE_' + fileOut + '.hdf5')
encoder = load_model('../Cl_data/Model/Encoder_' + fileOut + '.hdf5')
decoder = load_model('../Cl_data/Model/Decoder_' + fileOut + '.hdf5')
history = np.load('../Cl_data/Model/TrainingHistory_'+fileOut+'.npy')



plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs =  history[0,:]
    train_loss = history[1,:]
    val_loss = history[2,:]


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax.set_title('Loss')
    ax.legend(['train loss','val loss'])
    plt.tight_layout()
    plt.savefig('../Cl_data/Plots/Training_loss.png')

plt.show()



# ------------------------------------------------------------------------------
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# ------------------------------------------------------------------------------

PlotSampleID = np.arange(2) #[1, 10]
ErrTh = 5
PlotRatio = True
if PlotRatio:
    ls = np.load('../Cl_data/Data/Latinls_' + str(TestFiles) + '.npy')[2:]
    PkOriginal = np.load('../Cl_data/Data/LatinCl_'+str(TestFiles)+'.npy')[:,2:] # Original
    RealParaArray = np.load('../Cl_data/Data/LatinPara5_'+str(TestFiles)+'.npy')

    for i in range(np.shape(RealParaArray)[0]):

        RealPara = RealParaArray[i]

        RealPara[0] = rescale01(np.min(X1), np.max(X1), RealPara[0])
        RealPara[1] = rescale01(np.min(X2), np.max(X2), RealPara[1])
        RealPara[2] = rescale01(np.min(X3), np.max(X3), RealPara[2])
        RealPara[3] = rescale01(np.min(X4), np.max(X4), RealPara[3])
        RealPara[4] = rescale01(np.min(X5), np.max(X5), RealPara[4])

        test_pts = RealPara[:5].reshape(5, -1).T

        # ------------------------------------------------------------------------------

        W_pred = np.array([np.zeros(shape=latent_dim)])
        gp = {}
        for j in range(latent_dim):
            gp["fit{0}".format(j)] = george.GP(kernel)
            gp["fit{0}".format(j)].compute(XY[:, 0, :].T)
            W_pred[:, j] = gp["fit{0}".format(j)].predict(y[j], test_pts)[0]

        # ------------------------------------------------------------------------------

        x_decoded = decoder.predict(W_pred)


        plt.figure(94, figsize=(8,6))
        plt.title('Autoencoder+GP fit')
        cl_ratio = normFactor*x_decoded[0]/PkOriginal[i]
        relError = 100*np.abs(cl_ratio - 1)

        plt.plot(ls, cl_ratio, alpha=.5, lw = 1.0)

        # plt.xscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$C_l^{GPAE}$/$C_l^{Original}$')
        # plt.legend()
        plt.tight_layout()



        if i in PlotSampleID:


            plt.figure(99, figsize=(8,6))
            plt.title('Autoencoder+GP fit')
            # plt.plot(ls, normFactor * x_test[::].T, 'gray', alpha=0.1)

            plt.plot(ls, normFactor*x_decoded[0], 'r--', alpha= 0.5, lw = 1, label = 'emulated')
            plt.plot(ls, PkOriginal[i], 'b--', alpha=0.5, lw = 1, label = 'original')

            # plt.xscale('log')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$C_l$')
            plt.legend()
            # plt.tight_layout()

            plt.plot(ls[relError > ErrTh], normFactor*x_decoded[0][relError > ErrTh], 'gx', alpha=0.2, label='bad eggs', markersize = '3')
            plt.savefig('../Cl_data/Plots/GP_AE_output.png')


        print(i, 'ERR0R min max (per cent):', np.array([(relError).min(), (relError).max()]) )


    plt.axhline(y=1, ls='-.', lw=1.5)
    plt.savefig('../Cl_data/Plots/GP_AE_ratio.png')

    plt.show()


# ------------------------------------------------------------------------------


from model import *
from datasets import *
from utils_model import *


def executions():

    distance_type = 'pseudo-chordal'  # geodesic or  pseudo-chordal
    dataname = 'reuters-8'        # 'reuters-8' , 'newsgroups20', arxiv-4, housing, 'article-8', 'movie-review', hyperpartisan
    num_of_epochs = 100
    subspace_dim = 30
    localized = False
    balanced = False
    nprotos = 1
    sigma = 100
    act_fun = 'sigmoid' #'sigmoid', 'identity'

    # Note:
    # 1) for chordal distance, the learning rate (for w and r) should be smaller than the case in geodesic distance
    # 2) the smaller d is the smaller lr should be

    lr_w = 0.1  # reuters8 (10d) 0.01
    lr_r = 0.00001  # imdb20d: 0.1, 0.000001


    ## Load dataset
    Xtrain, Ytrain, Xval, Yval = load_data(dataname)
    if Xtrain.shape[-1] != subspace_dim:
        Xtrain = Xtrain[:, :, :subspace_dim]
        Xval = Xval[:, :, :subspace_dim]
    # Xval, Yval = Xtrain, Ytrain

    print(f"\nThere are '{Xtrain.shape[0]}' training and '{Xval.shape[0]}' testing examples on the manifold G({Xtrain.shape[-2]}, {Xtrain.shape[-1]}).")
    print(Xtrain.shape, Xval.shape)
    # ************** build the model **************
    # Note: for chordal distance, the learning rate (for w and r) should be
    # bigger than the case in geodesic distance
    print('\nConstructing the model ...')
    model = Model(
        dim_of_data=Xtrain.shape[-2],        # the dimensionality of data
        dim_of_subspace=Xtrain.shape[-1],    # number of data in a set
        num_of_classes=len(np.unique(Ytrain)),  # number of classes
        distance=distance_type,     # (pseudo) chordal or geodesic
        balanced=balanced,
        localized=localized,
        nprotos=nprotos,          # for now it is only 1: check if you need to modify it for more
        actfun=act_fun,# the function inside cost function: identity or sigmoid
        sigma=sigma,            # parameter for sigmoid function
    )

    # ******* initialize prototypes ***********
    print('Initializing prototypes ...')
    # via samples
    # model.initialize_parameters(xtrain=Xtrain, ytrain=Ytrain)
    # via normal distributions
    model.initialize_parameters(classes=Ytrain)
    print(model.xprotos.shape)

    # ************** fit the model **************
    acc_train = np.zeros(num_of_epochs + 1)
    acc_val = np.zeros(num_of_epochs + 1)

    pred = model.predict(Xtrain)
    acc_train[0], conf_mat_tr = model.metrics(Ytrain, pred)
    if Xval.size != 0:
        pred = model.predict(Xval)
        acc_val[0], conf_mat_val = model.metrics(Yval, pred)
        print("epoch {}: \t training accuracy: {:.2f}, \t testing accuracy: {:.2f} (max: {:.5f})".format(
            0, acc_train[0], acc_val[0], acc_val[0]))
    else:
        print("epoch {}: \t accuracy: {:.2f}".format(
            0, acc_train[0]))
    np.set_printoptions(precision=1)
    print(conf_mat_tr)

    fname = "../model/%s/%s_model_d%i_%s" % (
        dataname,
        dataname,
        # "localized_" if localized else "",
        Xtrain.shape[-1],
        distance_type[:2]
    )
    print('Fitting the model ...')
    for epoch in range(1, num_of_epochs+1):
        model.fit(
            Xtrain, Ytrain,
            lr_w=lr_w, lr_r=lr_r,
        )
        if epoch % 10 == 0:
            lr_w *= 0.5

        pred_tr = model.predict(Xtrain)
        acc_train[epoch], conf_mat_tr = model.metrics(Ytrain, pred_tr)
        pred_val = model.predict(Xval)
        acc_val[epoch], conf_mat_val = model.metrics(Yval, pred_val)

        print("relevances: ", model.lamda)
    #    print("epoch {}: \t training accuracy: {:.2f}, \t testing accuracy: {:.2f} (max: {:.5f})".format(
    #        epoch, acc_train[epoch], acc_val[epoch], np.max(acc_val[:epoch+1])))

        # np.set_printoptions(precision=1)
        # print(conf_mat_val)

      #  if epoch % 10 == 0:
      #      errorcurves(acc_train=acc_train[:epoch+1], acc_val=acc_val[:epoch+1], lamda=model.lamda)
      #      print(f"save model in: %s" % fname)
      #      model.save_results(fname, acc_train[:epoch+1], acc_val[:epoch+1], conf_mat_val)


if __name__ == '__main__':
    executions()

import numpy as np
import time # used to help me do training

# With an nvidia gpu one can do some sweet time optimisations.
# if cuda is available, 'cupy' will be used instead of 'numpy' for any of my matrix multiplications.
# https://docs.cupy.dev/en/stable/index.html
# With the fashion-MNIST dataset and a tesla K80 
# the per-epoch training time was ~24x quicker than that of a base 2021 mac book pro.

# I've already tried vectorising things in numpy as a first type of vectorisation, and that
# improved per-batch training speeds from 5 seconds to 2. 
# This was done by vectorising per-batch forward and backward passes along with derivative calculations
# instead of doing a for-loop over each item in the batch.
# However, this would have meant that per net training (40 epochs) would have taken around 6ish hours. 
# I find that too slow.
# I apologise for not completely technically following instructions about just using numpy.
# I find this replacement to 'cupy' to be reasonable, however, since I developed the entire code
# in numpy first, and it ran perfectly in numpy. In fact, if all pre-pendings of 'np' are
# changed to 'np' below, everything will train as usual and be the same, just slowly.
# I hope this is OK!

# Further, on choice of optimisation
# I used a line-profiler to find which functions were slow, and spent my time optimising them.
# Some redundancies exist, like function having other functions as inputs, but this either
# didn't cross my mind whilst looking at the line-profiler, or I did it on purpose to create
# nice figures used whilst developing code.



# I will define an object which acts as a single layer to the MLP.
# It has attributes W, B, last, each signifying the layer's weights
# biases, and whether or not it's the last layer.
# In the code below I implicitely consider feed-forward one-layer-wide neural nets
# This layer object has pre activation and post activation methods, allowing for
# either to be executed with respect to some input vector x.


class nn_hidden_layer:
    # assume PReLU as standard (p = 0.01)
    def __init__(self, W, B, last=False):
        self.W = W
        self.B = B
        self.last = last

    def activation(self, x):
        if not self.last:
            # applies PReLU
            x[x<0] = 0.01*x[x<0]
            return x
        else:
            # applies softmax
            exps = np.exp(x)
            return exps / np.sum(exps, axis=0)

    def pre_activation(self, x):
        # applies linear unit
        # cutoffs are used to make sure things don't diverge to infinity
        self.W = np.where(self.W>10, 10, self.W)
        self.B = np.where(self.B>10, 10, self.B)
        return np.squeeze(self.W@(x) + self.B)

    def post_activation(self, x):
        # applies required activation function
        return np.squeeze(self.activation(self.pre_activation(x)))


def softmax(x):
    #shifted_x = x - np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)

    
def sigma_dash(x):
    x = np.where(x>=0, 1, 0.01)
    return x


def crossentropy(x, y):
    # expects input of softmax output x
    # one-hot encoded true classification y
    # a safe crossentropy is used to prevent log 0
    xc = np.copy(x)
    xc = np.where(xc<0.0001, 0.0001, xc)
    xc = np.where(xc>0.9999, 1.0, xc)
    return -1*np.sum(y*(np.log(xc).T), axis=1)


def fwd_pass(crossentropy, layers, x, y=None):
    # the forward pass to the nn, this can but won't be optimised
    # x can be batched data, too!
    xc = np.copy(x).T
    pre_activations = []
    post_activations = []
    fetch_ws = []
    outputs=[xc]

    for layer in layers:
        pre_activations.append(layer.pre_activation(xc)) # for an entire batch
        post_activations.append(layer.post_activation(xc)) # for an entire batch
        outputs.append(layer.post_activation(xc)) # for an entire batch
        xc = layer.post_activation(xc) # for an entire batch
        
        fetch_ws.append(layer.W) # just the weights of the net

    if y is not None:
        loss = crossentropy(post_activations[-1], y) # for an entire batch
        return pre_activations, post_activations, outputs, fetch_ws, loss

    pred = np.argmax(outputs[-1], axis=0) # for an entire batch
    
    # outputs:
    ## pre_activations for all nodes in a network
    ## post_activations for all nodes in a network
    ## outputs for all nodes (this includes the input nodes as outputs)
    ## the weight matrices at each layer of a network
    ## pred: prediction given input
    return pre_activations, post_activations, outputs, fetch_ws, pred


def fast_predict(layers, x):
    xc = np.copy(x).T
    for layer in layers:
        xc = layer.post_activation(xc)
    
    return np.argmax(xc, axis=0)


def fast_predict_dataset(layers, x_data, y_data):
    y_hat = fast_predict(layers, x_data)
    yvals = np.argmax(y_data, axis=1)
    return np.sum(y_hat == yvals)/len(yvals)


# I use the general method found in the notes, which first computes delta values for each node
# and then computed bias and weight derivatives from this delta.
def delta_pass(pre_activations, post_activations, Ws, label, softmax, sigma_dash): # make v
    # derivative of softmax+crossentropy across batch
    delta_list = [post_activations[-1] - np.squeeze(label).T]
    t = len(pre_activations)-1
    # across each layer, find the delta values for each batch
    for delta in range(t):
        delta_list.append(sigma_dash(pre_activations[t-delta-1])*((Ws[t-delta].T)@(delta_list[delta])))
    
    return delta_list[::-1] # returns it in reverse order, so front to back of net


def derivatives(delta_pass, delta_list, outputs):
    weight_derivatives = []
    bias_derivatives = []
    # compute weight and bias derivatives for each batch item
    for i in range(len(delta_list)): # for each layer
        # does outer product for each batch item across batches
        # this finds the average weight derivatives over a batch in a layer
        np_outputs = np.asarray(outputs[i].T)
        np_deltas = np.asarray(delta_list[i].T)
        layers_weights_cupy = np.einsum('bi,bo->bio', np_outputs, np_deltas)
        layers_weights_cupy = np.transpose(layers_weights_cupy, (1, 2, 0))
        weight_derivatives.append(np.sum(layers_weights_cupy, axis=2)/layers_weights_cupy.shape[2])
        
        # average bias derivatives
        bias_derivatives.append(np.sum(delta_list[i], axis=1)/delta_list[i].shape[1])

    return weight_derivatives, bias_derivatives


# a simple update procedure based on constant step-sizes
def update_parameters(layers, weight_derivatives, bias_derivatives, stepsize, bwd_pass):
    n = len(layers)
    for layer in range(n):
        layers[layer].W -= (stepsize*weight_derivatives[layer]).T
        layers[layer].B -= (stepsize*np.expand_dims(bias_derivatives[layer], axis=1))
    return None


def bwd_pass(layers, pre_activations, post_activations, outputs, Ws, label, delta_pass, derivatives):
    # computes a list of deltas for the net, each item of the list corresponds
    # to a layer in the net, and each column is the delta for each batch item
    delta_list = delta_pass(pre_activations, post_activations, Ws, label, softmax, sigma_dash)
    # computes average weight and bias derivatives over the batch of the network
    weight_derivatives, bias_derivatives = derivatives(delta_pass, delta_list, outputs)
    return weight_derivatives, bias_derivatives


# this function does the forward pass, backwards pass, and gradient descent once.
def train_one_loop(layers, x, y, stepsize, fwd_pass, bwd_pass):
    # pre activations, post activations, outputs per layer
    # as arrays of numpy arrays, each column is for each layer
    # each numpy array is for each batch
    # loss for an entire batch is given as a numpy array
    # weights (ws) are given for the network at that iteration that the batch forward passess
    pre_act, post_act, outs, ws, loss = fwd_pass(crossentropy, layers, x, y)
    # finds weight_derivatives and bias_derivatives given by the avg of them over the batch
    weight_derivatives, bias_derivatives = bwd_pass(layers, pre_act, post_act, outs, ws, y, delta_pass, derivatives)
    # updates the weights and biases of the net
    update_parameters(layers, weight_derivatives, bias_derivatives, stepsize, bwd_pass)
    
    ## computing metrics
    # average cross-entropy loss over the batch
    loss = np.sum(loss)/len(loss)
    
    return loss


def make_W(in_size, out_size):
    # weights are initialised by kaiming initialisation
    # this is superior to the usual Xavier in our case.
    # effectively, the usual one was developed according to smooth
    # activation functions. Once rectified activation functions
    # became a thing, the maths no longer checked out. Kaiming et al.
    # re-did the maths on rectified functions, producing this initialising.
    # https://arxiv.org/pdf/1502.01852v1.pdf
    arr = np.random.normal(loc=0.0, scale= np.sqrt(2/in_size), size=(out_size, in_size))
    return np.asarray(arr)


def make_B(out_size):
    # biases initialised at 0
    arr = np.zeros(shape=(out_size, 1))
    return np.asarray(arr)


# this function loads the data into GPU memory to be used via CuPy.
# the batches are made by sampling randomly from the training set
# sampling almost as many items as there are in the training set, so
# the lists returns can be considered as epoch training sets.
def data_loader(x_train, y_train, no_batches, batch_size):
    # flatten input for each image
    x_train.reshape(*x_train.shape[:-2], -1)
    
    n = len(x_train)
    indices = {i for i in range(n)}
    xbatches = []
    ybatches = []
    for i in range(no_batches):
        p = np.random.choice(list(indices), size=batch_size, replace=False)
        data_indices = np.asarray(p)
        to_take_x = np.take(x_train, data_indices, axis=0)
        to_take_y = np.take(y_train, data_indices, axis=0)
        xbatches.append(to_take_x)
        ybatches.append(to_take_y)
        indices.difference_update(set(p))
    
    return xbatches, ybatches


def train_epoch(derivatives, data_loader, layers, x_train, y_train, x_valid, y_valid, step_size=10e-3, batch_size=256, predict=False):
    batches_per_epoch = len(y_train)//(batch_size)
    x_batches, y_batches = data_loader(x_train, y_train, batches_per_epoch, batch_size)
    losses = []
    
    progress_bar_length = 30
    for i in range(batches_per_epoch):
        losses.append(train_one_loop(layers, x_batches[i], y_batches[i], step_size, fwd_pass, bwd_pass))
        print(batches_per_epoch, 'batches per epoch: [', '.'*((i+1)//(batches_per_epoch//progress_bar_length)), end='\r')
    
    aggr_loss = np.sum(np.asarray(losses)/len(losses))
    print(batches_per_epoch, 'batches per epoch: [', '.'*progress_bar_length, '] loss: ', "%.4f"%aggr_loss, end='')
    if predict is True:
        train_accs = fast_predict_dataset(layers, x_train, y_train)
        val_accs = fast_predict_dataset(layers, x_valid, y_valid)
        print(' | train acc.: ', "%.4f"%train_accs, end=' | ')
        print('val acc.: ', "%.4f"%val_accs, end=' | ')
    
        avg_loss = np.sum(np.array(losses)/len(losses))
        return avg_loss, train_accs, val_accs
    elif predict is False:
        train_accs = fast_predict_dataset(layers, x_train, y_train)
        avg_loss = np.sum(np.array(losses)/len(losses))
        return avg_loss, train_accs


def train_nn(train_epoch, layers, x_train, y_train, x_valid, y_valid, no_epochs, step_size=10e-3, batch_size=256, predict=False):
    losses = []
    training_accs = []
    if predict:
        validation_accs = []
    
    print('\n', '!!! STARTED TRAINING !!!', end = '\n'*2)
    print('NET TYPE: ', 'FEED FORWARD NET') # add a property when doing this to specify neural network type
    print('EPOCHS: ', no_epochs)
    print('TRAINING INPUT SIZE: ', x_train.shape, ', LABEL INPUT SIZE: ', y_train.shape)
    if predict == True:
        print('VALIDATION INPUT SIZE: ', x_valid.shape, ', LABEL INPUT SIZE: ', y_valid.shape)
    print('UPDATE PROCEDURE: SGD')
    print('LOSS: L2')
    print('STEP SIZE: ', step_size, ', BATCH SIZE: ', batch_size, end='\n'*2)

    t0 = time.perf_counter()
    for i in range(no_epochs):
        trained_outputs = train_epoch(derivatives, data_loader, 
                                 layers, x_train, y_train, x_valid, y_valid,
                                 step_size=step_size, batch_size=batch_size, predict=predict)
        losses.append(trained_outputs[0])
        training_accs.append(trained_outputs[1])
        if predict:
            validation_accs.append(trained_outputs[1])
        print(i+1, ' out of ', no_epochs, ' epochs')

    t1 = time.perf_counter()
    print('\n', '!!! DONE TRAINING!!!', end='\n'*2)
    print('ELAPSED TIME:', "%.1f"%(t1-t0), 'SECONDS.')
    print('PER EPOCH:', "%.1f"%(((t1-t0)/no_epochs)), 'SECONDS.', end='\n'*2)
    print('RETURNS: (layers, losses, training_accs, validation_accs)')

    if predict:
        return layers, losses, training_accs, validation_accs
    else:
        return layers, losses, training_accs


class feedforwardnet:
    def feed_1d(self, x, y):
        if len(x) != len(y):
            print("Make sure both x and y datasets either have or haven't gotten validation splits")
        if len(x) == 2 and len(y) == 2:
            self.x_train = np.expand_dims(x[0], -1)
            self.y_train = np.expand_dims(y[0], -1)
            self.x_test = np.expand_dims(x[1], -1)
            self.y_test = np.expand_dims(y[1], -1)
            self.predict = True
        if len(x) == 1 and len(y) == 1:
            self.x_train = np.expand_dims(x[0], -1)
            self.y_train = np.expand_dims(y[0], -1)
            self.predict = False

    def feed_2d(self, x, y):
        if len(x) != len(y):
            print("Make sure both x and y datasets either have or haven't gotten validation splits")
        
        if len(x) == 2 and len(y) == 2:
            x_train_copy = np.reshape(x[0], (x[0].shape[0], x[0].shape[1]*x[0].shape[2]))
            x_test_copy = np.reshape(x[1], (x[1].shape[0], x[1].shape[1]*x[1].shape[2]))
            self.x_train = x_train_copy
            self.y_train = y[0]
            self.x_test = x_test_copy
            self.y_test = y[1]
            self.predict = True
        if len(x) == 1 and len(y) == 1:
            x_train_copy = np.reshape(x[0], (x[0].shape[0], x[0].shape[1]*x[0].shape[2]))
            self.x_train = x_train_copy
            self.y_train = y[0]
            self.predict = False

    def make_layers(self, layers, verbose=False):
        make_layers = []
        last = False
        for layer in range(len(layers)-1):
            if layer == len(layers) - 2:
                last = True
            
            make_layers.append(nn_hidden_layer(make_W(layers[layer], layers[layer+1]), make_B(layers[layer+1]), last))
        self.untrained_net = make_layers
        
        if verbose:
            print('Initialised Feed-forward net layers with "he" initialisation.', end = '\n'*2)
            print('Input size: {}'.format(layers[0]))
            print('Output size: {}'.format(layers[-1]))
            print('Passing through fully-connected layers of width: {}'.format(layers[1:-1]))

    def train(self, no_epochs=10, step_size=10e-3, batch_size=32):
        if self.predict:
            trained_layers, trained_losses, trained_accs, test_accs = train_nn(train_epoch, self.untrained_net, 
                                                                               self.x_train, self.y_train, self.x_test, self.y_test, 
                                                                               no_epochs=no_epochs, 
                                                                               step_size=step_size, 
                                                                               batch_size=batch_size, 
                                                                               predict=self.predict)
            self.history = [trained_losses, trained_accs, test_accs]
            self.trained_net = trained_layers
        else:
            trained_layers, trained_losses, trained_accs = train_nn(train_epoch, self.untrained_net, 
                                                                    self.x_train, self.y_train, self.x_train, self.y_train, 
                                                                    no_epochs=no_epochs, 
                                                                    step_size=step_size, 
                                                                    batch_size=batch_size, 
                                                                    predict=self.predict)
            self.history = [trained_losses, trained_accs]
            self.trained_net = trained_layers
        print('ACCESS trained net as (feedforwardnet).trained_net, and loss history as (feedforwardnet).history')
        print('ACCESS layer weight and biases in net as (feedforwardnet).trained_net[layer_num].W, (feedforwardnet).trained_net[layer_num].B')
    
    def predict(self, x):
        return fast_predict(self.trained_net, x)
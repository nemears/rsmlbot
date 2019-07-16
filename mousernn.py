import lasagne
import theano.tensor as T
import theano
import numpy as np
import csv

def open_csv(path):
    with open(path+'.csv','r',newline='') as file:
        reader = csv.reader(file)
        x = []
        for row in reader:
            x.append([float(row[0])/773,float(row[1])/534])
        for p in x:
            p.append(x[-1][0])
            p.append(x[-1][1])
        return np.array(x)

def build_mouse_model(seq_size):
    in_var = T.tensor3('input')
    tar_var = T.matrix('target')
    network = {}
    N_LSTM = 4 #control number of LSTM units
    network['input'] = lasagne.layers.InputLayer(shape=(None,seq_size,4),
                                                 input_var = in_var)
    n_batch,n_step,n_feature = network['input'].input_var.shape
    network['LSTM'] = lasagne.layers.LSTMLayer(network['input'],N_LSTM,
                                               nonlinearity = lasagne.nonlinearities.tanh,
                                               grad_clipping=100)
    #squash sequence dimensions
    network['reshape'] = lasagne.layers.ReshapeLayer(network['LSTM'],(n_batch,N_LSTM*seq_size))
    network['output'] = lasagne.layers.DenseLayer(network['reshape'],2,
                                              nonlinearity = lasagne.nonlinearities.tanh)
    prediction = lasagne.layers.get_output(network['output'])
    cost = T.mean((prediction-tar_var)**2)
    params = lasagne.layers.get_all_params(network['output'],trainable=True)
    updates = lasagne.updates.adam(cost,params)
    train = theano.function([in_var,tar_var],cost,updates=updates)
    cost_val = theano.function([in_var,tar_var],cost)
    prediction = theano.function([in_var],prediction)
    return network, train, cost_val,prediction

def sequential_data(x,jump):
    x_out,y_out = [],[]
    for i in range(len(x)-jump-1):
        x_seq = np.zeros((jump,np.shape(x)[-1]))
        for j in range(jump):
            x_seq[j] = x[i+j]
        x_out.append(x_seq)
        y_out.append([x[i+jump+1][0],x[i+jump+1][1]])
    return x_out,y_out

def train_model(trainf,valf,x,jump,x_val=None):
    print("training...")
    for epoch in range(100):
        x_d, y_d = sequential_data(x,jump)
        loss = trainf(x_d,y_d)
        print("Epoch: {}, loss: {}".format(epoch,loss))
    if x_val is not None:
        x_d_val, y_d_val = sequential_data(x_val,jump)
        val = valf(x_d_val,y_d_val)
        print('validation loss:',val)

if __name__ == '__main__':
    network, train, cost_val, prediction = build_mouse_model(5)
    x_val = open_csv('tospot30')
    for i in range(1,30):
        x = open_csv('tospot'+str(i))
        train_model(train,cost_val,x,5,x_val)
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network['output']))
    

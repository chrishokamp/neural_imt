"""
The confidence model asks how sure we are that our next prediction will be correct

Training input for this model is (source, prefix, CORRECT \in {0,1}), where the 1/0 label
is obtained by comparing the prediction of the current IMT model to the reference.

The model outputs a single scalar which can be interpreted as p(correct(w_t))

IMPLEMENTATION:
Pass the initial states through the Readout to get the prediction

The sequence generator for this model is the same one we are using to generate the data

The cost function is binary crossentropy for (batch, prediction) -- i.e (batch, 1)

binary cross entropy http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.binary_crossentropy

At inference time, if we pass along the state for every model in the beam search, we can also get the model's confidence at each time step




"""




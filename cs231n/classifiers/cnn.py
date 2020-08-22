from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        (C, H, W) = input_dim
        conv_dim = int(num_filters*H/2*W/2)

        self.params = {
            'W1': weight_scale*np.random.randn(num_filters, C, filter_size, filter_size), 
            'b1': np.zeros([num_filters, ], dtype = float), 
            'W2': weight_scale*np.random.randn(conv_dim, hidden_dim),
            'b2': np.zeros([hidden_dim, ], dtype = float),
            'W3': weight_scale*np.random.randn(hidden_dim, num_classes),
            'b3': np.zeros([num_classes, ], dtype = float),
        }
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        out_conv, cache_conv = conv_forward_fast(X, W1, b1, conv_param)
        out_mp, cache_mp = max_pool_forward_fast(out_conv, pool_param)
        out_hidden, cache_hidden =affine_relu_forward(out_mp, W2, b2)
        out_affine, cache_affine = affine_forward(out_hidden, W3, b3)

        scores = out_affine
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss = loss + self.reg*0.5*(np.sum(np.power(self.params['W1'],2))+np.sum(np.power(self.params['W2'],2))+np.sum(np.power(self.params['W3'],2)))
        # dx = dx + self.reg * (np.sum(self.params['W1'])+np.sum(self.params['W2'],2))

        dx3, grads['W3'], grads['b3'] = affine_backward(dx, cache_affine)
        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, cache_hidden)
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        dx_mp = max_pool_backward_fast(dx2, cache_mp)
        dx1, grads['W1'], grads['b1'] = conv_backward_fast(dx_mp, cache_conv)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

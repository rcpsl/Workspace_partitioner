import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        # NOTE: image size should be twice of number of lasers 

        # Test 1
        """
        self.image_size = 5
        fc1_size = 4
        fc2_size = 3
        fc3_size = 3
        fc4_size = 2
        self.nn_size = [fc1_size, fc2_size, fc3_size, fc4_size]
        W1 = np.array([[3.0, -4.0, 5.0, 10.0], [-7.0, -2.0, 5.0, 0.0], [0.0, -3.0, 1.0, 2.0], [-3.0, -2.0, 1.0, 0.0], [4.0, 3.0, 2.0, 1.0]])
        b1 = np.array([3.0, -2.0, -1.0, 0.0]).reshape((1, fc1_size))
        W2 = np.array([[2.0, 3.0, -5.0], [6.0, 3.0, -7.0], [-2.0, 0.0, 3.0], [1.0, 2.0, 1.0]])
        b2 = np.array([-1.0, -3.0, 2.0]).reshape((1, fc2_size))
        W3 = np.array([[-3.0, 0.0, 3.0], [2.0, 1.0, 2.0], [1.0, 5.0, 2.0]])
        b3 = np.array([0.0, 3.0, 2.0]).reshape((1, fc3_size))
        W4 = np.array([[-1.0, 3.0], [5.0, -2.0], [4.0, 0.0]])
        b4 = np.array([-12.0, -1.0]).reshape((1, fc4_size))
        self.W = [W1, W2, W3, W4]
        self.b = [b1, b2, b3, b4]
        """

        # Test 2
        """
        self.image_size = 8
        fc1_size = 3
        fc2_size = 2
        fc3_size = 2
        self.nn_size = [fc1_size, fc2_size, fc3_size]

        W1 = np.array([[3.0, 5.0, 6.0], [-11.0, 3.0, 0.0], [0.0, -7.0, 3.0], [-2.0, 6.0, 7.0], 
                       [9.0, 7.0, 2.0], [4.0, -4.0, -5.0], [-2.0, 1.0, -6.0], [0.0, 1.0, -1.0]])
        b1 = np.array([-10.0, -3.0, 5.0]).reshape((1, fc1_size))
        W2 = np.array([[-6.0, 0.0], [5.0, 4.0], [-1.0, 3.0]])
        b2 = np.array([0.0, -3.0]).reshape((1, fc2_size))
        W3 = np.array([[1.0, 0.0], [0.0, 1.0]])
        b3 = np.array([0.0, 0.0]).reshape(1, fc3_size)
        self.W = [W1, W2, W3]
        self.b = [b1, b2, b3]
        """

        # Test 3
        self.image_size = 16
        fc1_size = 9
        fc2_size = 8
        fc3_size = 7
        fc4_size = 2
        self.nn_size = [fc1_size, fc2_size, fc3_size, fc4_size]

        W1 = np.random.randn(self.image_size, fc1_size)
        b1 = np.random.randn(fc1_size).reshape((1, fc1_size))
        W2 = np.random.randn(fc1_size, fc2_size)
        b2 = np.random.randn(fc2_size).reshape((1, fc2_size))
        W3 = np.random.randn(fc2_size, fc3_size)
        b3 = np.random.randn(fc3_size).reshape((1, fc3_size))
        W4 = np.random.randn(fc3_size, fc4_size)
        b4 = np.random.randn(fc4_size).reshape((1, fc4_size))
        self.W = [W1, W2, W3, W4]
        self.b = [b1, b2, b3, b4]


        # Last layer does not have ReLUs
        self.num_relus = sum(self.nn_size) - self.nn_size[-1]



    def fc2matrix(self, relus):
        """
        Compute parameters in following two constraints based on trained NN weights and ReLU assignments
        - Dependence of output y on input image x0: y = x0 G + h
        - Constraint of ReLU assignment on input images: x0 Q <= c
        
        Inputs:
        - relus: a list of True and False
        """
        G, h, Q, c = None, None, None, None
        #relus = [True, False, True, True, True, True, False, False, True, True] # Test 1
        #relus = [True, False, True, True, True] # Test 2
        relus = np.array([int(x) for x in relus])

        # Split ReLU assigments based on layers
        split_indices = self.nn_size[:-2]
        split_indices = np.cumsum(split_indices)
        relus = np.split(relus, split_indices)
        #print 'relus = ', relus

        # Need to flip signs based on ReLU assignments when compute Q and c
        Q_signs, c_signs = [], []
        for relus_this_layer in relus:
            layer_size = len(relus_this_layer)
            Q_signs_this_layer = np.ones((layer_size, ), dtype=int)
            c_signs_this_layer = np.ones((layer_size, ), dtype=int)
            Q_signs_this_layer[relus_this_layer == 1] = -1 
            c_signs_this_layer[relus_this_layer == 0] = -1
            Q_signs.append(Q_signs_this_layer)
            c_signs.append(c_signs_this_layer)

        # From first to second last layer of NN
        G = np.identity(self.image_size)
        h = np.zeros((1, self.image_size))
        for i in xrange(len(relus)):
            G = np.dot(G, self.W[i])
            h = np.dot(h, self.W[i]) + self.b[i]
            Q = Q_signs[i] * G
            c = c_signs[i] * h
            G *= relus[i]
            h *= relus[i]
            if i == 0:
                Q_tot = Q
                c_tot = c
            else:     
                Q_tot = np.concatenate((Q_tot, Q), axis=1)
                c_tot = np.concatenate((c_tot, c), axis=1)

        # Last layer does not have ReLUs
        G = np.dot(G, self.W[-1])
        h = np.dot(h, self.W[-1]) + self.b[-1]

        # NN convention is row vector, while routines that generate constraints assumes column vector
        GT = np.transpose(G)
        h  = h.tolist()[0]
        QT = np.transpose(Q_tot)
        c  = c_tot.tolist()[0]

        return GT, h, QT, c

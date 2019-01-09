import numpy as np

class NeuralNetwork(object):

    def __init__(self):
        # NOTE: image size should be twice of number of lasers
        self.image_size = 5
        self.fc1_size   = 4
        self.fc2_size   = 3
        self.fc3_size   = 3
        self.fc4_size   = 2

        self.W1 = np.array([[3.0, -4.0, 5.0, 10.0], [-7.0, -2.0, 5.0, 0.0], [0.0, -3.0, 1.0, 2.0], [-3.0, -2.0, 1.0, 0.0], [4.0, 3.0, 2.0, 1.0]])
        self.b1 = np.array([3.0, -2.0, -1.0, 0.0]).reshape((1, self.fc1_size))
        self.W2 = np.array([[2.0, 3.0, -5.0], [6.0, 3.0, -7.0], [-2.0, 0.0, 3.0], [1.0, 2.0, 1.0]])
        self.b2 = np.array([-1.0, -3.0, 2.0]).reshape((1, self.fc2_size))
        self.W3 = np.array([[-3.0, 0.0, 3.0], [2.0, 1.0, 2.0], [1.0, 5.0, 2.0]])
        self.b3 = np.array([0.0, 3.0, 2.0]).reshape((1, self.fc3_size))
        self.W4 = np.array([[-1.0, 3.0], [5.0, -2.0], [4.0, 0.0]])
        self.b4 = np.array([-12.0, -1.0]).reshape((1, self.fc4_size))

        # No relu for the last layer
        self.num_relus = self.fc1_size + self.fc2_size + self.fc3_size



    def fc2matrix(self, relus):
        """
        Compute parameters in following two constraints for given trained NN and ReLU assignments
        - Acceptable input images x0 need to satisfy ReLU assignments: x0 * Q <= c
        - Output y depends on input image x0: y = x0 * G + h

        Inputs:
        - relus: a list of True and False

        TODO: flexibly change number of ReLUs and layers.
        """
        # TODO: better slice indices
        relus = np.array([int(x) for x in relus])
        print 'relus = ', relus
        s1 = relus[:self.fc1_size]
        s2 = relus[self.fc1_size: self.fc1_size+self.fc2_size]
        s3 = relus[self.fc1_size+self.fc2_size:]
        #print s1
        #print s2
        #print s3
        s1 = np.array([1, 0, 1, 1])
        s2 = np.array([1, 1, 0])
        s3 = np.array([0, 1, 1])

        # Flip signs based on ReLU assignments 
        sp1 = s1.copy()
        sz1 = s1.copy()
        sp2 = s2.copy()
        sz2 = s2.copy()
        sp3 = s3.copy()
        sz3 = s3.copy()
        sp1[s1 == 1] = -1 
        sp1[s1 == 0] = 1
        sz1[s1 == 0] = -1
        sp2[s2 == 1] = -1
        sp2[s2 == 0] = 1
        sz2[s2 == 0] = -1
        sp3[s3 == 1] = -1
        sp3[s3 == 0] = 1
        sz3[s3 == 0] = -1
    
        # 1st layer (fc + ReLU)
        G1 = self.W1
        h1 = self.b1
        Q1 = sp1 * G1 # Flip signs of columns that associate to ReLU=1
        c1 = sz1 * h1 # Flip signs of elements that associate to ReLU=0 
        G1 *= s1
        h1 *= s1
        # 2nd layer (fc + ReLU)
        G2 = np.dot(G1, self.W2)
        h2 = np.dot(h1, self.W2) + self.b2
        Q2 = sp2 * G2
        c2 = sz2 * h2 
        G2 *= s2
        h2 *= s2
        # 3rd layer (fc + ReLU)
        G3 = np.dot(G2, self.W3)
        h3 = np.dot(h2, self.W3) + self.b3
        Q3 = sp3 * G3
        c3 = sz3 * h3 
        G3 *= s3
        h3 *= s3
        # 4th layer (fc only)
        G = np.dot(G3, self.W4)
        h = np.dot(h3, self.W4) + self.b4

        Q = np.concatenate((Q1, Q2, Q3), axis=1)
        c = np.concatenate((c1, c2, c3), axis=1) 

        return Q, c, G, h

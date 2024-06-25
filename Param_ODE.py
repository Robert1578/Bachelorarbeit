import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')         

import matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import time
import colorama
from matplotlib import animation

#Import the DiffEq class
from Param_Diff_Eq import DiffEq

#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

#tensorFlow accuracy
tf.keras.backend.set_floatx('float64')


#Custom plot fontsize
import os
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin/'
from matplotlib import cm
from matplotlib import rc
plt.rcParams['axes.labelsize'] = 15                                                                                                     
plt.rcParams['legend.fontsize'] = 10                                                                                                     
plt.rcParams['xtick.labelsize'] = 10                                                                                                     
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"                                                                                                   
plt.rcParams['font.serif'] = "cm"


class Param_ODEsolver():
    
    def __init__(self, order, diffeq, x,git, initial_condition, epochs, architecture, initializer, activation, optimizer,*args):
        """
        order : differential equation order (ex: order = 2) 
        diffeq : differential equation as defined in the class DiffEq
        x : training domain (higher dimensional input (x,λ,...) for parameter dependet ODE)
        initial_condition : initial condition including x0 and y0 (ex: initial_condition = (x0 = 0, y0 = 1))
        architecture : number of nodes in hidden layers (ex: architecture = [10, 10])
        initializer : weight initializer (ex: 'GlorotNormal')
        activation : activation function (ex: tf.nn.sigmoid)
        optimizer : minimization optimizer including parameters (ex: tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, beta_2 = 0.5, epsilon = 1e-07))
        prediciton_save : bool to save predicitons at each epoch during training (ex: prediction_save = False)
        weights_save : bool to save the weights at each epoch (ex: weights_save = True)
        """
        colorama.init()
        self.GREEN = colorama.Fore.GREEN
        self.RESET = colorama.Fore.RESET
        tf.keras.backend.set_floatx('float64')
        self.order = order
        self.diffeq = diffeq
        self.x = x
        self.initial_condition = initial_condition
        self.n = len(self.x)                             # Anzahl Gitterpunkte
        self.d = len(self.x[0])                          # Dimension des Inputs
        self.epochs = epochs
        self.architecture = architecture
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.neural_net = self.build_model()    # self.neural_net_model(show = True)
        self.neural_net.summary()
        self.git=git                                   #Anzahl Gitterpunkte pro Dimension als Liste
        

        #Compile the model
        x = self.x
        x = tf.convert_to_tensor(x)                     
        x = tf.reshape(x, (self.n, self.d) )            
        #Alternativ tf.keras.losses.mse self.custom_cost(x)
        self.neural_net.compile(loss = self.custom_cost(x) , optimizer = self.optimizer)      

        
        
        
    def build_model(self):
        """
        Builds a customized neural network model.
        """
        architecture = self.architecture
        initializer = self.initializer
        activation = self.activation
        
        nb_hidden_layers = len(architecture)
        input_tensor = tf.keras.layers.Input(shape = (self.d,))          
        hidden_layers = []


        if nb_hidden_layers >= 1:
            hidden_layer = tf.keras.layers.Dense(architecture[0], kernel_initializer= initializer, bias_initializer='zeros',activation = activation)(input_tensor)
            hidden_layers.append(hidden_layer)
            for i in range(1, nb_hidden_layers):
                hidden_layer = tf.keras.layers.Dense(architecture[i], kernel_initializer= initializer, bias_initializer='zeros',activation = activation)(hidden_layers[i-1])
                hidden_layers.append(hidden_layer)
            output_layer = tf.keras.layers.Dense(1, kernel_initializer= initializer, bias_initializer = 'zeros', activation = tf.identity)(hidden_layers[-1])
        else:
            output_layer = tf.keras.layers.Dense(1, kernel_initializer= initializer, bias_initializer = 'zeros', activation = tf.identity)(input_tensor)
        
        model = tf.keras.Model(inputs = input_tensor, outputs = output_layer)
        return model
    
    
    @tf.function                #decorator
    def NN_output(self, x):
        """
        x : must be of shape = (?, self.d)
        Returns the output of the neural net
        """
        y = self.neural_net(x)
        return y
    

    def y_gradients(self, x):

        """
        Computes the gradient of y. Symbolic Differentiation
        Recording Operations: When you use with tf.GradientTape()as tape:
        you create a context manager that tracks all the mathematical operations performed within that block.
        Symbolic Representation: These operations are internally represented as a computational graph.
        Gradient Calculation: When you call tape.gradient(y, x), where y is the output and x is the variable you 
        want the gradient with respect to, the tape analyzes the computational graph. It uses the symbolic 
        representation to calculate the partial derivative of y with respect to x.
        """

        with tf.GradientTape() as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y = self.NN_output(x)
            dy_dx = tape2.gradient(y, x)
        d2y_dx2 = tape1.gradient(dy_dx,x)
        return y, dy_dx, d2y_dx2

    
    def differential_cost(self, x):

        """
        Defines the differential cost function for one neural network input.
        """
                                                                 
        y, dydx, d2ydx2 = self.y_gradients(x)       #y output vom NN, Dim(dydx)=d
        
        #----------------------------------------------
        #------------DIFFERENTIAL-EQUATION-------------
        #----------------------------------------------
        #print(y.shape)
        #print(x.shape)

        de = DiffEq(self.diffeq, x, y, dydx, d2ydx2,)
        differential_equation = de.eq

        #----------------------------------------------
        #----------------------------------------------
        #----------------------------------------------
        
        return tf.square(differential_equation)





    @tf.function                #decorator
    def maxx(self,x):
        y=tf.maximum(tf.math.log(x)**2, 0)
        return y

    def continuity_cost(self, x):           # hinzugefügt um seltsame Funktionen vom NN zu vermeiden
        continuity_cost_term = 0
        for i in range(self.git[0]-2):
            delta=tf.divide(self.NN_output(x)[(i)*self.git[1]:(i+1)*self.git[1]],self.NN_output(x)[(i+1)*self.git[1]:(i+2)*self.git[1]])
            #if np.max(delta.numpy())<0.5:#tf.constant(2, shape=(1, 1), dtype=tf.float64):
            continuity_cost_term = tf.maximum(tf.reduce_sum(tf.reduce_max(tf.math.abs(delta))), continuity_cost_term)
        return self.maxx(continuity_cost_term )






    def custom_cost(self, x):
        """
        Defines the cost function for a batch.     loss function (5) in der Quelle
        """
        if self.order == 1:

            def loss(y_true, y_pred):                                               
                differential_cost_term = tf.math.reduce_sum(self.differential_cost(x))
                continuity_cost_term = self.continuity_cost(x)


                if self.initial_condition != False:
                    x0 = self.initial_condition[0]
                    y0 = self.initial_condition[1]
                    xa=self.x[self.x[:, 0] == 0]            # alle Einträge der Form [0,λ] für Randbedingung    
                    boundary_cost_term = tf.math.reduce_sum(tf.square(self.NN_output(xa)-y0))    # -y0 funktioniert komponentenweise
                    return differential_cost_term/self.n + boundary_cost_term   #+ continuity_cost_term
                else: 
                    xa=self.x[self.x[:, 0] == 0]           # alle Einträge der Form [0,λ,u0] für Randbedingung                      
                    boundary_cost_term = tf.math.reduce_sum(tf.square(self.NN_output(xa)-tf.reshape(xa[:,2],(len(xa),1))))    
                    return differential_cost_term/self.n + boundary_cost_term      
            return loss
        # order == 2 noch nicht angefangen
        '''
        if self.order == 2:
            x0 = np.float64(self.initial_condition[0][0])
            y0 = np.float64(self.initial_condition[0][1])
            dx0 = np.float64(self.initial_condition[1][0])
            dy0 = np.float64(self.initial_condition[1][1])

            def loss(y_true, y_pred):
                differential_cost_term = tf.math.reduce_sum(self.differential_cost(x))
                boundary_cost_term = tf.square(self.NN_output(np.asarray([[x0]]))[0][0] - y0)
                boundary_cost_term += tf.square(self.NN_output(np.asarray([[dx0]]))[0][0] - dy0)
                return differential_cost_term/self.n + boundary_cost_term
            return loss
        '''
    
    
    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to x.
        """
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, self.d) )           #x = tf.reshape(x, (self.n, 1))     # 2-D Tensor (self.n) rows, 1 column 
        

        neural_net = self.neural_net
        
        #Train and save the predicitons deleted, because not relevant
        #Train without any saving   

        start_time = time.time()
        #If x and y were the same, it wouldn't be meaningful learning. The model wouldn't be learning relationships 
        #between features and target variables; it would simply be memorizing the data itself.
        #Da die loss Funktion in .compile die exakte Lösung nicht braucht ist das möglich
        history = neural_net.fit(x = x, y = x, batch_size = self.n, epochs = self.epochs)
        f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time)
        print(f"{self.RESET}")
            
        return history
    
    
    def get_loss(self, history):
        """
        history : history of the training procedure returned by self.train
        Returns epochs and loss
        """
        epochs = history.epoch
        loss = history.history["loss"]
        return epochs, loss
    
    
    def predict(self, x_predict):
        """
        x_predict : domain of prediction 
        """
        domain_length = len(x_predict)
        x_predict = tf.convert_to_tensor(x_predict)
        x = tf.reshape(x_predict, (domain_length, self.d))
        y_predict = self.neural_net.predict(x)

        return y_predict


    def relative_error(self, y_predict, y_exact):
        """
        y_predict : array of predicted solution
        y_exact : array of exact solution
        Returns the relative error of the neural network solution
        given the exact solution.
        """
        #print(y_exact.shape)
        #print(y_predict.shape)
        if len(y_exact) != len(y_predict):
            raise Exception("y_predict and y_exact do not have the same shape.")
        relative_error = np.abs(y_exact - np.reshape(y_predict, (self.n)))/np.abs(y_exact)
        return relative_error


    def mean_relative_error(self, y_predict, y_exact):
        """
        y_predict : array of predicted solution
        y_exact : array of exact solution
        Returns the single mean relative error value of 
        the neural network solution given the exact solution.
        """
        relative_error = self.relative_error(y_predict, y_exact)
        relative_error = relative_error[relative_error < 1E100]
        return np.mean(relative_error)


    def absolute_error(self, y_predict, y_exact):
        """
        y_predict : array of predicted solution
        y_exact : array of exact solution
        Returns the mean absolute error of the neural network solution
        given the exact solution.
        """
        if len(y_exact) != len(y_predict):
            raise Exception("y_predict and y_exact do not have the same shape.")
        absolute_error = np.abs(y_exact - np.reshape(y_predict, (100)))
        return absolute_error
    

   
        

    

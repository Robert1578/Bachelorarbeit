import tensorflow as tf
import numpy as np
from Param_ODE import Param_ODEsolver
from scipy.optimize import minimize

class opti():
    
    
    def __init__(self,NN_instance, Data, initial_guess,bounds):
        self.NN_instance=NN_instance
        self.param=len(initial_guess)
        self.Data=Data
        self.initial_guess = initial_guess
        self.bounds=bounds
    

#--------------------------------------------------------------------
#----------------------gradient descent------------------------------
#--------------------------------------------------------------------
    
    def optimize_gradient_descent(self,learning_rate):
        history=[]
        losslist=[]
        p=tf.Variable(self.initial_guess)
        Data=tf.convert_to_tensor(self.Data)
        i=0

        def objective(t,p,y):
            p=tf.tile(tf.reshape(p,(1,self.param)),multiples=(len(Data),1)) # Damit concat funktioniert (5,1), (5,2)
            t=tf.reshape(t,(len(t),1))                                  
            x=tf.concat([t,p],axis=1)    
            fNN=self.NN_instance.NN_output(x) #tf.shape,(len(t),1)) #davor NN_output(x)
            y=tf.cast(y, dtype=tf.float64)
            y=tf.reshape(y,(len(y),1))
            return tf.reduce_sum(tf.square(fNN-y))        

        while 1:                      #for _ in range(3000):  # Replace with a stopping criteria
            with tf.GradientTape() as tape:
                loss = objective(tf.constant(Data[:,0]),p,tf.constant(Data[:,1]))
            dp = tape.gradient(loss, p)                #davor [p])[0]
            p.assign_sub(learning_rate* dp)
            losslist.append(np.squeeze(loss.numpy()))
            history.append(np.squeeze(p.numpy()))#np.squeeze(p.numpy()))
            i=i+1
            if loss<10**(-9):
                break
            elif i>1499:
                break
        return p.numpy(), losslist, history


#--------------------------------------------------------------------
#-------------------------L-BFGS-------------------------------------
#--------------------------------------------------------------------

    def optimize_LBFGS(self):
        
        def objective2(parameter,Data):
            squared_errors = [(self.NN_instance.predict(np.append([Data[t][0]], parameter).T.reshape(-1, self.param + 1)) - Data[t][-1])**2 for t in range(len(Data))]
            return tf.reduce_sum(squared_errors)
        
        p = minimize(objective2,self.initial_guess, args=(self.Data),bounds=self.bounds)   
                                          #parameter?
        return p.x   

# davor np.array(np.meshgrid())


#--------------------------------------------------------------------
#---------------------------ADAM-------------------------------------
#--------------------------------------------------------------------

    def optimize_Adam(self,learning_rate):
        history=[]
        losslist=[]
        p=tf.Variable(self.initial_guess)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        Data=tf.convert_to_tensor(self.Data)
        i=0

        def objective(t,p,y):
            p=tf.tile(tf.reshape(p,(1,self.param)),multiples=(len(Data),1))    
            t=tf.reshape(t,(len(t),1))              # Damit concat funktioniert   
            x=tf.concat([t,p],axis=1)    
            fNN=self.NN_instance.NN_output(x) #tf.shape,(len(t),1)) 
            y=tf.cast(y, dtype=tf.float64)
            y=tf.reshape(y,(len(y),1))
            return tf.reduce_sum(tf.square(fNN-y))        

        while 1:                      #for _ in range(3000):  # Replace with a stopping criteria
            with tf.GradientTape() as tape:
                loss = objective(tf.constant(Data[:,0]),p,tf.constant(Data[:,1]))
            dp = tape.gradient(loss, p)                #davor [p])[0]
            optimizer.apply_gradients(zip([dp], [p]))
            losslist.append(np.squeeze(loss.numpy()))
            history.append(np.squeeze(p.numpy()))
            i=i+1
            if loss<0.000001:
                break
            elif i>1000:
                break
        return p.numpy(), losslist, history

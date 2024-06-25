import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch     						    # save the weights of a neural network
import math

#Import the Dictionary class
from Dict import Dictionary
D1 = Dictionary();	D4 = Dictionary();	D7 = Dictionary()
D2 = Dictionary();	D5 = Dictionary();	D8 = Dictionary()									#Änderung
D3 = Dictionary();	D6 = Dictionary();	D9 = Dictionary()
Dict,Dict2,Dict3,Dict4,Dict5,Dict6,Dict7,Dict8,Dict9 = D1.Dict,D2.Dict,D3.Dict,D4.Dict,D5.Dict,D6.Dict,D7.Dict,D8.Dict,D9.Dict

from scipy.optimize import minimize

#Import the DiffEq class
from Param_Diff_Eq import DiffEq

#import parameter dependet ODE solver
from Param_ODE import Param_ODEsolver

#import Optimizer
from NN_Parameter_Optimizer import opti

#Random seed initialization
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)


#--------------------------------------------------------------------
#-----------------------------u' = λu--------------------------------
#--------------------------------------------------------------------


#----------------------------Input = 2D------------------------------

# Differential equation exp(λt)
order = 1
diffeq = "trivialbsp"
# Training domain
t = np.linspace(0, 1, 11)
λ = np.linspace(0, 1, 11)
x_2D = np.array(np.meshgrid(t, λ)).T.reshape(-1, 2)		
# Gitter t x λ
git=[len(t),len(λ)]
# Initial conditions for ODE-solver
initial_condition = (0, 1) 
# Number of epochs
epochs = 40000
# Structure of the neural net (only hidden layers)
architecture = [40,20]
# Initializer used
initializer = Dict["initializer"]["GlorotNormal"]
# Activation function used
activation = Dict["activation"]["sigmoid"]
# Optimizer used
optimizer = Dict["optimizer"]["Adam"]

solver = Param_ODEsolver(order, diffeq, x_2D,git, initial_condition, epochs, architecture, initializer, activation, optimizer)


# Training
history = solver.train()
epoch, lossNN = solver.get_loss(history)


# Trainingswerte
x_predict = x_2D
y_predict = solver.predict(x_predict)
y_exact = 	tf.math.exp(x_predict[:,0]*x_predict[:,1])	
print(solver.mean_relative_error(y_predict, y_exact))


# 3D plot of the NN approximation
y_predict= np.array(y_predict)
y_predict=y_predict.reshape((len(t), len(λ)))
y_exact= np.array(y_exact)
y_exact=y_exact.reshape((len(t), len(λ)))
λ, t= np.array(np.meshgrid(λ, t))

ax = plt.axes(projection='3d')
ax.contour3D(t,λ, y_predict, 100, cmap='cividis')
# ax.set_title('NN, exp($\lambda$t)')
ax.set_xlabel('t')
ax.set_ylabel("$\lambda$")
ax.set_zlabel('$NN(t,\lambda)  \\approx exp(t\lambda)$')
	
ax.view_init(60, 35)
plt.tight_layout()


# plotting the NN approximation using pcolormesh
z = np.array(y_predict)
z=z.reshape((git[0], git[1]))
figpcolor = plt.figure()
plt.pcolormesh(λ, t, z, cmap="plasma")
plt.colorbar(label="Color values")  # Add colorbar
plt.xlabel("t")
plt.ylabel("$\lambda$")
plt.title("NN, exp($\lambda$t)")


# plot loss/Epoch of the NN
figloss = plt.figure(figsize = (9.5, 4.6))
axloss = plt.axes() 
axloss.set_xlim(epoch[100], epoch[-1])
#ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
#ax2.set_xlim(epoch[0], epoch[-1])
axloss.set_xticks([10000, 20000, 30000, 40000])                                        # Epochen-Schritte für x-Achse
axloss.set_xticklabels(["$10^4$", "$2.10^4$", "$3.10^4$", "$4.10^4$"])
axloss.set_xlabel("Anzahl der Epochen", fontsize = 15)
figloss.text(0.02, 0.5, "Fehlerfunktion $\mathcal{L}(\Theta)$", fontsize = 15, va = 'center', rotation='vertical')

axloss.semilogy(epoch, lossNN, label = "Architektur = (40,20)", color = "C4")
axloss.legend()

#ax2.semilogy(epoch, loss, label = "(20, 20, 20, 20)", color = "C3")
#ax2.legend()

#----------------------Optimizer-2D----------------------------------

# data points [[t1, f1],[t2, f2],...]
Data=[[0.7,math.exp(0.49)],[0.2,math.exp(0.14)],[0.1,math.exp(0.07)],[0.5,math.exp(0.35)],[1,math.exp(0.7)]]
# NN instance to differentiate the neural network during optimization
NN_instance=solver
# inital guess for the parameter(vector)
initial_guess = [0.4]
# learning rate for gradient descent
learning_rate=0.001
# bounds of the optimization for L-BFGS
bounds=[(0,1)]



Optimierung=opti(NN_instance, Data, initial_guess,bounds)


# Gradient Descent
y,losslist,history=Optimierung.optimize_gradient_descent(learning_rate)


lossrelative = [abs(x/0.7-1) for x in history]      # real λ=0.7


# plot Optimization 
figlossopti,axlossopti = plt.subplots(figsize = (9.5, 4.6))
ax2lossopti = axlossopti.twinx()
axlossopti.set_xlim(0, len(losslist))
ax2lossopti.plot(epoch[:len(losslist)], losslist,'b-')
axlossopti.plot(epoch[:len(losslist)], lossrelative,'g-')

#ax2lossopti.set_xlim(epoch[0], epoch[-1])
axlossopti.set_xticks([300,600,900,1200,1500])                                        # Epochen-Schritte für x-Achse
axlossopti.set_xticklabels(["$300$", "$600$","$900$","$1200$", "$1500$" ])
axlossopti.set_xlabel("Iterationen", fontsize = 15)

ax2lossopti.set_ylabel('$\lambda$: Optimierungsfehler ',color='b')
axlossopti.set_ylabel('$\lambda$: realer relativer Fehler',color='g')
ax2lossopti.semilogy()
axlossopti.semilogy()
#axlossopti.legend()



# L-BFGS
#z=Optimierung.optimize_LBFGS()
#print(z)


# ADAM
#z,losslistADAM,historyADAM=Optimierung.optimize_Adam(learningrate=0.01)
#print(z)

plt.show()



#----------------------------Input = 3D------------------------------

'''
#Differential equation exp(λt)
order = 1
diffeq = "trivialbsp"
#Training domain
t = np.linspace(0, 1, 8)
λ = np.linspace(0, 1, 8)
u0= np.linspace(0.5, 1.5, 8)											
x_3D = np.array(np.meshgrid(t, λ, u0)).T.reshape(-1, 3)	#np.meshgrid(t, λ, [1])
#Gitter t x λ x u0
git=[len(t),len(λ),len(u0)] 
#Initial conditions for ODE-solver
initial_condition = False		# u0=x[0,λ,u0]=N(0,λ,u0)
#Number of epochs
epochs = 100000
#Structure of the neural net (only hidden layers)
architecture = [50,20]						# 20x20x20 besser als 30x30
#Initializer used
initializer = Dict["initializer"]["GlorotNormal"]
#Activation function used
activation = Dict["activation"]["sigmoid"]
#Optimizer used
optimizer = Dict["optimizer"]["Adam"]

	
solver = Param_ODEsolver(order, diffeq, x_3D,git, initial_condition, epochs, architecture, initializer, activation, optimizer)

#Training
history = solver.train()
epoch, loss = solver.get_loss(history)

x=[[0,1,0.9],[0,1,1],[0,1,1.1],[0,0.5,0.9],[0,0.5,1],[0,0.5,1.1],
   [0.5,0.5,0.9],[0.5,0.5,1],[0.5,0.5,1.1],[1,0.5,0.9],[1,0.5,1],[1,0.5,1.1],
   [1,1,0.9],[1,1,1],[1,1,1.1]]
print(solver.neural_net(tf.convert_to_tensor(x)))


x_predict = x_3D
y_predict = solver.predict(x_predict)
y_exact = 	tf.multiply(x_predict[:,2],tf.math.exp(x_predict[:,0]*x_predict[:,1]))
	
print(solver.mean_relative_error(y_predict, y_exact))
'''


# plotting 3D Wie soll das überhaupt klappen
'''
x_predict = x_3D
y_predict = solver.predict(x_predict)
y_exact = 	tf.multiply(x_predict[:,2],tf.math.exp(x_predict[:,0]*x_predict[:,1]))
	
#print(solver.mean_relative_error(y_predict, y_exact))

# plotting
λ, t, u0= np.array(np.meshgrid(λ, t, u0))
y_predict= np.array(y_predict)
y_predict=y_predict.reshape((len(t), len(λ)))
y_exact= np.array(y_exact)
y_exact=y_exact.reshape((len(t), len(λ)))

ax = plt.axes(projection='3d')
ax.contour3D(t,λ, y_predict, 100, cmap='cividis')
ax.set_title('NN, exp($\lambda$t)')
ax.set_xlabel('t')
ax.set_ylabel("$\lambda$")
ax.set_zlabel('NN(t,$\lambda$)')
	
ax.view_init(60, 35)
plt.tight_layout()
plt.show()
'''

#plotting using pcolormesh
'''
λ, t = np.array(np.meshgrid(λ, t))
z = np.array(y_exact)
z=z.reshape((len(t), len(λ)))

plt.pcolormesh(λ, t, z, cmap="plasma")
plt.colorbar(label="Color values")  # Add colorbar
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Pseudocolor plot with pcolormesh")
plt.show()
'''
#----------------------Optimizer-3D----------------------------------

'''
# data points [[t1, f1],[t2, f2],...]
u_0=1.2
Data=[[0.7,u_0*math.exp(0.35)],[0.2,u_0*math.exp(0.1)],[0.1,u_0*math.exp(0.05)],[0.5,u_0*math.exp(0.25)],[1,u_0*math.exp(0.5)],
    [0.3,u_0*math.exp(0.15)],[0.4,u_0*math.exp(0.2)],[0.6,u_0*math.exp(0.3)],[0.8,u_0*math.exp(0.4)],[0.9,u_0*math.exp(0.45)]]

# NN instance to differentiate the neural network during optimization
NN_instance=solver
# inital guess for the parameter 
initial_guess = [0.5,1]
# bounds of the optimization for L-BFGS
bounds=[(0,1),(0.5,1.5)]
# learning rate for gradient descent; vector, because it might depend on the parameter


Optimierung=opti(NN_instance, Data, initial_guess,bounds)



x,losslist=Optimierung.optimize_gradient_descent(0.001)	# funktioniert nicht richtig konvergiert bei (0.78, 0.8) etwa
print(x)
print(losslist)
print()

y=Optimierung.optimize_LBFGS()
print(y)
print()

z,loss2=Optimierung.optimize_Adam(0.01)
print(z)
print(loss2)
'''

#plot loss for gradient descent
'''
figlossopti,axlossopti = plt.subplots(figsize = (9.5, 4.6))
ax2lossopti = axlossopti.twinx()
axlossopti.set_xlim(0, len(losslist))
ax2lossopti.plot(epoch[:len(losslist)], losslist,'b-')
axlossopti.plot(epoch[:len(losslist)], lossrelative,'g-')

#ax2lossopti.set_xlim(epoch[0], epoch[-1])
axlossopti.set_xticks([300,600,900,1200,1500])                                        # Epochen-Schritte für x-Achse
axlossopti.set_xticklabels(["$300$", "$600$","$900$","$1200$", "$1500$" ])
axlossopti.set_xlabel("Iterationen", fontsize = 15)

ax2lossopti.set_ylabel('$\lambda$: Optimierungsfehler ',color='b')
axlossopti.set_ylabel('$\lambda$: realer relativer Fehler',color='g')
ax2lossopti.semilogy()
axlossopti.semilogy()
#axlossopti.legend()
'''






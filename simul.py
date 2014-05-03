__author__ = 'teofilo'


from math import *
import pendulum
import rnn
import logger

# ip params 
theta = None
thetaprime = None
x = None
xprime = None
good_enough = None

theta_tolerance = 0.1
thetaprime_tolerance = 0.1
xprime_tolerance = 0.1

# rnn params
n_in = 4
n_hidden = 25 
n_out = 1

learning_rate = 0.001 
learning_rate_decay = 0.001

n_epochs = 200
activation = 'tanh'

def update_ip_vars(ip):
    global theta, thetaprime, x, xprime

    theta = ip.rotational[0]
    thetaprime = ip.rotational[1]
    x = ip.translational[0]
    xprime = ip.translational[1]

def update_good_enough(ip):
    update_ip_vars(ip)

    return (xprime < xprime_tolerance) and \
     (thetaprime < thetaprime_tolerance) and \
     (abs(theta - 180) < theta_tolerance)

if __name__ == "__main__":
	ip = pendulum.InvertedPendulum

	our_rnn = rnn.MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                    n_epochs=n_epochs, activation=activation,
                    output_type='softmax', use_symbolic_softmax=False)

    update_good_enough(ip)

    while not good_enough:

        our_rnn.fit(X_test=, 
				Y_test=, 
				validation_frequency=100)

        controlinput = our_rnn.predict()
        
        ip.update(controlinput)

    	update_good_enough(ip)
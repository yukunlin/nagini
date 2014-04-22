##Nagini: (N)eur(a)l networks controller for a (g)enerated (i)(n)verted pendulum (i)nterface


###Background

The task of a controller is to determine and manipulate the inputs of a system in a way such that the output of that system matches with some desired outputs. For the purpose of maintaining a stable desired output, PID (proportional, integral, derivative) controllers are typically used.

###Problem Statement
<p align="center">
  <img src="https://raw.githubusercontent.com/yukunlin/nagini/master/pid.gif">
</p>

Balancing an inverted pendulum is a classic control problem.

###Approach

Our goal is to design a recurrent neural network (RNN) that is able perform as well as (or better) than a PID controller at balancing the inverted pendulum.

We will do this through an RNN implementation in Theano.

###Reference

http://deeplearning.net/tutorial/DBN.html
http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf
http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf
http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf
http://www.iro.umontreal.ca/~lisa/pointeurs/theano_scipy2010.pdf
https://www.stanford.edu/class/ee373b/NNselflearningcontrolsystems.pdf
http://research.cs.wisc.edu/machine-learning/shavlik-group/scott.NeuralComp92.pdf
http://archive.ics.uci.edu/ml/index.html

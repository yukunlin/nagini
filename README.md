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



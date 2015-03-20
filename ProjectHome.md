## Warning ##

This software is not usable for now, I just pushed the raw source code of a quick tool that I made for my experiments. I plane to reengineer it soon.

## General informations ##

This program is a command-line tool that provides an easy way of learning, testing and running [artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) (ANN). It uses the [Fast Artificial Neural Network library](http://leenissen.dk/fann/) (FANN) for learning and testing.

### What are artificial neural networks ? ###
An artificial neural network (ANN), usually called "neural network" (NN), is a mathematical model or computational model that tries to simulate the structure and/or functional aspects of biological neural networks. An ANN performs a non-parametric non-linear multivariate multiple regression.

### What is Fast Artificial Neural Network library ? ###
The Fast Artificial Neural Network library (FANN) is a free open source neural network library, which implements multilayer artificial neural networks in C and supports both fully and sparsely connected networks. Cross-platform execution in both fixed and floating point is supported. It includes a framework for easy handling of training data sets. It is easy to use, versatile, well documented, and fast. PHP, C++, .NET, Ada, Python, Delphi, Octave, Ruby, Prolog Pure Data and Mathematica bindings are available.

## News ##

  * 2010.05.08: support of the _Continuous_ and _Labels_ Icsiboost data fields (option -S)
  * 2010.04.19: initial release

## Features of the program ##

The main features of this program are :
  * Support of [Icsiboost](http://code.google.com/p/icsiboost) / [Boostexter](http://www.cs.princeton.edu/~schapire/boostexter.html) file format (see [ProgramUsage](http://code.google.com/p/sfann/wiki/ProgramUsage))
  * Support of the native Fann data format
  * Designed for resolving classification problems, error measurement is done in terms of classification error rate
  * Automatic cross-validation evaluation
  * Auto-saving of ANN that performs the best on train, dev or test

Please take a look at the [ProgramUsage](http://code.google.com/p/sfann/wiki/ProgramUsage) page for more details.

## TODO - Not yet implemented (but should be soon) ##

  * support of the _Text_ Icsiboost data field
  * selection of the training algorithm (currently it uses RPROP)
  * selection of the number of layers (currently it produces 3 layer MLPs)
  * selection of activation functions (currently it uses symmetric sygmoids for both hidden and output layers)
  * selection of network density (currently it produces full connected MLPs)
  * selection of error mode (currently only classification mode)

## Getting and compiling ##

The verison 2.1.0 of FANN is needed in order to properly compile sfann.

```
svn checkout http://sfann.googlecode.com/svn/trunk/ sfann
cd sfann
./configure
make
```
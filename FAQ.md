### What does CCR and MSE stand for ? ###

In the context of MLP evaluation, MSE means _Mean Square Error_ and CCR means _Correct Classification Rate_. The lower the MSE is and the higher the CCR is, the better the MLP is. The measures are computed according to these formulas:

  * CCR = num\_exemples\_correctly\_classified / num\_exemples\_to\_classify
  * MSE = sum<sub>output_neurons</sub>((output\_neuron\_value-expected\_output\_neuron\_value)<sup>2</sup>)/num\_output\_neurons
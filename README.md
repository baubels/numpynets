Neural nets from scratch using NumPy.

![](gif_highres.gif)
### Usage
1) clone repo
2) install locally (`python -m pip install -e ./numpynets`)
3) get 2d image data 

`x_train.shape = (n, height, width), y_train.shape = (n, classes)` or in addition provide `x_valid.shape=(m, height, width), y_valid.shape(m, classes)`

4) initialise and run an arbitrary length feed-forward fully connected net

With the learnt net: 
1) extract learnt weights/biases with `.trained_ned[layer_num].W`, `.trained_net[layer_num].B`
2) extract learning histories/losses using `.history`
3) predict values for new inputs with `.predict(xdata)`

### Network Specs
* He initialisations
* (Stochastic) Gradient Descent
* Feed-forward and fully connected

### To implement
* convs
* 1d data as input (req's a minor bug fix)
* easier custom loss, activations, initialisations
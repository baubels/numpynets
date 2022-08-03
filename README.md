Building neural nets from scratch using NumPy. Allows for agile 2D classifications using arbitrary-length fully-connected feed-forward deep neural nets.

### Usage
1) clone repo
2) install locally (MacOS: `python -m pip install -e ./numpynets`)
3) get 2d image data 

`x_train.shape = (n, height, width), y_train.shape = (n, classes)` or in addition provide `x_valid.shape=(m, height, width), y_valid.shape(m, classes)`

4) initialise and run an arbitrary length feed-forward fully connected net

With the learnt net you can 
1) extract learnt weights/biases with `.trained_ned[layer_num].W`, `.trained_net[layer_num].B`
2) extract learning histories/losses
3) predict values for new inputs

### Example

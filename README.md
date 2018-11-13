# Python Neural Network

This code shows how to create a modular code for neural networks. It can be easily extended to create any kind of layers, like Convolutional Layers.

For now, this code only implements *Stochastic Gradient Descent* (SGD).

# Layers

#### Fully Connected Layer
`FCLayer(input_shape, output_shape)`

#### Activation Layer
`ActivationLayer(input_shape, activation, activation_prime)`

# Example

```python
# xor data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]]);
y_train = np.array([[[0]], [[1]], [[1]], [[0]]]);

# network
net = Network();
net.add(FCLayer((1,2), (1,3)));
net.add(ActivationLayer((1,3), act, act_prime));
net.add(FCLayer((1,3), (1,1)));
net.add(ActivationLayer((1,1), act, act_prime));

# train
net.use(loss, loss_prime);
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1);
```

# Numpy for algebraic operations
# NNFS for spiral data construction and weights initialization
# Matplotlib for visualization
# Time to measure training time
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
import time

# Initializes the weights and biases randomly
nnfs.init()

# Create data
X, y = spiral_data(samples=100, classes=3)

# Dense Layer
class Layer:

    # Weights' and biases' initialization
    def __init__(self, n_inputs, n_neurons,
                 l1_weight_strength=0, l2_weight_strength=0,
                 l1_bias_strength=0, l2_bias_strength=0):
        self.weights = 0.0001 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set strength of L1 L2 penalties
        self.l1_weight_strength = l1_weight_strength
        self.l2_weight_strength = l2_weight_strength
        self.l1_bias_strength = l1_bias_strength
        self.l2_bias_strength = l2_bias_strength

    # Foward Propagation
    def forprop(self, inputs, training):
        # Keep track of input values
        self.inputs = inputs

        # Calculate output from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward Pass
    def backprop(self, dvalues):
        # Weight gradient (multiplication rule: treat inputs as constants)
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Bias gradient (sum rule)
        self.dbiases = np.sum(dvalues,
                              axis=0,
                              keepdims=True)

        # Regularization gradients
        # L1 for weights (binary masks)
        if self.l1_weight_strength > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.l1_weight_strength * dL1
        # L2 weights
        if self.l2_weight_strength > 0:
            self.dweights += 2 * self.l2_weight_strength * self.weights
        # L1 for biases
        if self.l1_bias_strength > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.l1_bias_strength * dL1
        # L2 biases
        if self.l2_bias_strength > 0:
            self.dbiases += 2 * self.l2_bias_strength * self.biases

        # Value Gradients
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation function
class ReLU:

    # Forward propagation
    def forprop(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backpropagation
    def backprop(self, dvalues):
        # Since variables are passed by reference in Python, we copy the values
        self.dinputs = dvalues.copy()

        # Gradient is zero for negative inputs
        self.dinputs[self.inputs <= 0] = 0

    # ReLU's predictions are simply the outputs themselves
    def predictions(self, outputs):
        return outputs


# Sigmoid activation function
class Sigmoid:

    def forprop(self, inputs, training):
        self.inputs = inputs

        self.output = 1. / (1. + np.exp(-inputs))

    def backprop(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate sigmoid predictions
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


# SoftMax activation function
class Softmax:

    def forprop(self, inputs, training):
        # Store input values
        self.inputs = inputs

        # Get unnormalized values that have been clipped by the
        #   maximum value to avoid overflow errors
        exp_values = np.exp(inputs - np.max(inputs,
                                            axis=1,
                                            keepdims=True))

        # Normalize values for each sample
        probabilities = exp_values / np.sum(exp_values,
                                            axis=1,
                                            keepdims=True)

        self.output = probabilities

    def backprop(self, dvalues):
        # Make uninitialized array with dvalues' dimensions
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate the Jacobian
            jacobian = np.diagflat(single_output) - \
                       np.dot(single_output, single_output.T)

            # Calculate the per-sample gradient and append it to the array
            #    of sample gradients
            self.dinputs[index] = np.dot(jacobian,
                                         single_dvalues)

    # Calculate predictions using the argmax function
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Softmax classifier - faster backward step when using composition
#   deliberately left unchecked to demonstrate overflowing parameters
class Softmax_Loss_Aggregate:

    # Backward pass
    def backprop(self, dvalues, y_true):
        # Sample number
        samples = len(dvalues)

        # One-hot encode if not already
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy values for safe processing
        self.dinputs = dvalues.copy()

        # Generate gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize
        self.dinputs = self.dinputs / samples


# Accuracy class
class Accuracy:

    def calculate(self, predictions, y):
        # Get comparisons
        comparisons = self.compare(predictions, y)

        # Calculate accuracy
        accuracy = np.mean(comparisons)

        # Return acc
        return accuracy


# Categorical Cross-Entropy Accuracy
class Accuracy_Categorical(Accuracy):

    # No need to initialize
    def init(self, y):
        pass

    # Compare predictions to the groun truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# General Loss class
class Loss:

    # Regularization loss
    def reg_loss(self):

        # By default let it be 0
        reg_loss = 0

        for layer in self.trainable_layers:

            # L1 reg for weights to be calculated when strength > 0
            if layer.l1_weight_strength > 0:
                reg_loss += layer.l1_weight_strength * \
                            np.sum(np.abs(layer.weights))

            # Same for L2 weights
            if layer.l2_weight_strength > 0:
                reg_loss += layer.l2_weight_strength * \
                            np.sum(layer.weights * layer.weights)

            # Same for biases now
            if layer.l1_bias_strength > 0:
                reg_loss += layer.l1_bias_strength * \
                            np.sum(np.abs(layer.biases))

            if layer.l2_bias_strength > 0:
                reg_loss += layer.l2_bias_strength * \
                            np.sum(layer.biases * layer.biases)

        return reg_loss

    # Store trainable layers for backprop with the data from the Model class
    def store_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculate data and regulaization loss
    #   given output of model and labels
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate loss of sample
        sample_losses = self.forprop(output, y)

        # Calculate the mean
        data_loss = np.mean(sample_losses)

        # If there is only data loss, no regularization loss
        if not include_regularization:
            return data_loss

        # Return it
        return data_loss, self.reg_loss()


# Categorical Cross-Entropy Loss
class CategoricalCrossEntropyLoss(Loss):

    # Forward pass
    def forprop(self, y_pred, y_true):

        # Number of samples in batch
        samples = len(y_pred)

        # Clip data to prevent div by 0
        # Clip data from both sides to prevent mean from being dragged towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values
        #   only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backprop(self, dvalues, y_true):

        # Sample number
        samples = len(dvalues)

        # Number of labels per sample, one sample is enough for that
        labels = len(dvalues[0])

        # If labels are sparse, one-hot encode
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradient
        self.dinputs = -y_true / dvalues

        # Normalize
        self.dinputs = self.dinputs / samples


# Optimizer
class Optimizer_Adam:

    # Initialize optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before parameters are updated
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # Parameter update
    def update_parameters(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases

        # Get correct momentum
        correct_weight_momentums = layer.weight_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))
        correct_bias_momentums = layer.bias_momentums / \
                                 (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache w/ squared gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases ** 2

        # Get correct cache
        correct_weight_cache = layer.weight_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))
        correct_bias_cache = layer.bias_cache / \
                             (1 - self.beta_2 ** (self.iterations + 1))

        # Traditional Stochastic Grad Descent paramter update and normalization
        # with cache under sqrt
        layer.weights += -self.current_learning_rate * \
                         correct_weight_momentums / \
                         (np.sqrt(correct_weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        correct_bias_momentums / \
                        (np.sqrt(correct_bias_cache) + self.epsilon)

    # Call once after update_parameters
    def post_update_parameters(self):
        self.iterations += 1

    # Dropout


class Layer_Dropout:

    def __init__(self, rate):
        # Invert given rate as per Bernoulli distribution
        self.rate = 1 - rate

    # Forward pass, similar to other classes'
    def forprop(self, inputs, training):
        # Save inputs
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to outputs
        self.output = inputs * self.binary_mask

    # Back pass
    def backprop(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Input layer
class Input_Layer:

    # Forward pass
    def forprop(self, inputs, training):
        self.output = inputs


# Full model class
class Model:

    # Layers' list, loss, optimizer and accuracy initialization
    def __init__(self):
        self.layers = []
        self.loss = CategoricalCrossEntropyLoss()
        self.optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)
        self.accuracy = Accuracy_Categorical()

    # Add layer object to layers list
    def add(self, layer):
        self.layers.append(layer)

    # Parameters' getter
    def get_parameters(self):
        return self.weights, self.biases
    
    # Complete the model
    def connect_layers(self):

        # Initialize an input layer
        self.input_layer = Input_Layer()

        # Count objects
        layer_count = len(self.layers)

        # Initialize list to contain trainable layers
        self.trainable_layers = []

        # Iterate through the objects and link them together
        for i in range(layer_count):

            # If this is the first layer in the layer_count list,
            #   the previous layer is the Input Layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # Last layer is the loss and save the last activation 
            #   function (softmax for this program)
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer has "weights" (or biases if one prefers, one is enough)
            #    attributes, then it is trainable
            #    so we add it to the list of trainable layers
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

            # Update loss object with trainable layers
            self.loss.store_trainable_layers(self.trainable_layers)

        # Create this object for faster gradient calculation
        self.softmax_classifier_output = Softmax_Loss_Aggregate()

    # Forward propagation
    def forprop(self, X, training):

        # Call all forprop methods in all the layers
        # We start with the Input Layer first
        self.input_layer.forprop(X, training)

        # Now all the hidden layers and the output which
        #   are included in the layers list
        for layer in self.layers:
            layer.forprop(layer.prev.output, training)

        if training is None:
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
            return


        # "layer" is the last element from the loop, so return
        #   its output since it's the output layer
        return layer.output

    # Backpropagation
    def backprop(self, output, y):

        # Trace backprop methods in order of the Chain Rule
        # First in line is the loss backprop
        self.softmax_classifier_output.backprop(output, y)

        self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

        # Call backprop methods going through all objects but last in reverse
        #   where dinputs of the next layer is the parameter to pass now
        for layer in reversed(self.layers[:-1]):
            layer.backprop(layer.next.dinputs)

    # Training
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize the accuracy object (if applicable)
        self.accuracy.init(y)

        # Primary training loop
        for epoch in range(1, epochs + 1):

            # Perform forward pass via forprops
            output = self.forprop(X, training=True)

            # Loss generation
            data_loss, reg_loss = self.loss.calculate(output, y,
                                                      include_regularization=True)
            loss = data_loss + reg_loss

            # Generate predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backpropagation
            self.backprop(output, y)

            # Optimize using the info from backpropagation
            self.optimizer.pre_update_parameters()
            for layer in self.trainable_layers:
                self.optimizer.update_parameters(layer)
            self.optimizer.post_update_parameters()

            # Print information
            if not epoch % print_every:
                print(f"epoch: {epoch}, " +
                      f"acc: {accuracy:.3f}, " +
                      f"loss: {loss:.3f} (" +
                      f"data_loss: {data_loss:.3f}, " +
                      f"reg_loss: {reg_loss:.3f}), " +
                      f"lr: {self.optimizer.current_learning_rate}")

        if validation_data is not None:
            # To discern between training and validation data
            X_val, y_val = validation_data

            # Forward pass
            output = self.forward(X_val, training=False)

            # Calculate loss
            loss = self.loss.calculate(output, y_val)

            # Compute predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')


# ReLU model
def model_run1():
   
    start_time = time.time()
    
    model1 = Model()     

    model1.add(Layer(2, 126, l2_weight_strength=5e-7, l2_bias_strength=5e-7))
    model1.add(ReLU())
    model1.add(Layer_Dropout(0.1))
    model1.add(Layer(126, 3))
    model1.add(Softmax())

    model1.connect_layers()

    pre_train_predictions = model1.forprop(X, training=False)

    # Train model
    model1.train(X, y, epochs=10000, print_every=100)

    # Training time
    execution_time = (time.time() - start_time) / 60
    print(f"--- {execution_time:.2f} minutes ---\n")
    
    post_train_predictions = model1.forprop(X, training=False)

    return pre_train_predictions, post_train_predictions

# Sigmoid with regularization
def model_run2():
    start_time = time.time()

    model2 = Model()

    model2.add(Layer(2, 126, l2_weight_strength=5e-7, l2_bias_strength=5e-7))
    model2.add(Sigmoid())
    model2.add(Layer_Dropout(0.1))
    model2.add(Layer(126, 3))
    model2.add(Softmax())

    model2.connect_layers()

    # Store pre-train predictions
    pre_train_predictions = model2.forprop(X, training=False)

    # Train model
    model2.train(X, y, epochs=10000, print_every=100)

    # Training time
    execution_time = (time.time() - start_time) / 60
    print(f"--- {execution_time:.2f} minutes ---\n")

    # Store pre-train predictions
    post_train_predictions = model2.forprop(X, training=False)

    return pre_train_predictions, post_train_predictions


# Sigmoid without regularization
def model_run3():
    start_time = time.time()

    model3 = Model()

    model3.add(Layer(2, 126))
    model3.add(Sigmoid())
    model3.add(Layer_Dropout(0.1))
    model3.add(Layer(126, 3))
    model3.add(Softmax())

    model3.connect_layers()

    # Store pre-train predictions
    pre_train_predictions = model3.forprop(X, training=False)

    # Train model
    model3.train(X, y, epochs=10000, print_every=100)

    # Training time
    execution_time = (time.time() - start_time) / 60
    print(f"--- {execution_time:.2f} minutes ---\n")
    
    # Store post-training predictions
    post_train_predictions = model3.forprop(X, training=False)

    # Return both sets of predictions
    return pre_train_predictions, post_train_predictions

# Create the rows on which the model names will be specified
def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

# Lists of predictions before and after training
pre_train_1, post_train_1 = model_run1()
pre_train_2, post_train_2 = model_run2()
pre_train_3, post_train_3 = model_run3()

# Create a figure with 9 subplots
rows = 3
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(9, 9))

# Name the three rows for the models
grid = plt.GridSpec(rows, cols)
create_subtitle(fig, grid[0, ::], 'ReLU Model')
create_subtitle(fig, grid[1, ::], 'Regulated Sigmoid Model')
create_subtitle(fig, grid[2, ::], 'Unregulated Sigmoid Model')
fig.tight_layout()
fig.set_facecolor('w')

# Plot the graphs on the figure

# Model 1
axs[0, 0].scatter(X[:, 0], X[:, 1], c=pre_train_1, cmap='brg')
axs[0, 0].title.set_text('Pre-training')
axs[0, 1].scatter(X[:, 0], X[:, 1], c=post_train_1, cmap='brg')
axs[0, 1].title.set_text('Post-training')
axs[0, 2].scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
axs[0, 2].title.set_text('Formal Classes')

# Model 2
axs[1, 0].scatter(X[:, 0], X[:, 1], c=pre_train_2, cmap='brg')
axs[1, 0].title.set_text('Pre-training')
axs[1, 1].scatter(X[:, 0], X[:, 1], c=post_train_2, cmap='brg')
axs[1, 1].title.set_text('Post-training')
axs[1, 2].scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
axs[1, 2].title.set_text('Formal Classes')

# Model 3
axs[2, 0].scatter(X[:, 0], X[:, 1], c=pre_train_3, cmap='brg')
axs[2, 0].title.set_text('Pre-training')
axs[2, 1].scatter(X[:, 0], X[:, 1], c=post_train_3, cmap='brg')
axs[2, 1].title.set_text('Post-training')
axs[2, 2].scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
axs[2, 2].title.set_text('Formal Classes')

# Show the plot
plt.show()
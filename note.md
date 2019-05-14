Getting the best model
To set your hyperparameters to get the best performance, you'll want to watch the training and validation losses. If your training loss is much lower than the validation loss, you're overfitting. Increase regularization (more dropout) or use a smaller network. If the training and validation losses are close, you're underfitting so you can increase the size of the network.

Hyperparameters
Here are the hyperparameters for the network.

In defining the model:

n_hidden - The number of units in the hidden layers.
n_layers - Number of hidden LSTM layers to use.
We assume that dropout probability and learning rate will be kept at the default, in this example.

And in training:

n_seqs - Number of sequences running through the network in one pass.
n_steps - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
lr - Learning rate for training
Here's some good advice from Andrej Karpathy on training the network. I'm going to copy it in here for your benefit, but also link to where it originally came from.

Tips and Tricks
Monitoring Validation Loss vs. Training Loss
If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

If your training loss is much lower than validation loss then this means the network might be overfitting. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
If your training/validation loss are about equal then your model is underfitting. Increase the size of your model (either number of layers or the raw number of neurons per layer)
Approximate number of parameters
The two most important parameters that control the model are n_hidden and n_layers. I would advise that you always use n_layers of either 2/3. The n_hidden can be adjusted based on how much data you have. The two important quantities to keep track of here are:

The number of parameters in your model. This is printed when you start training.
The size of your dataset. 1MB file is approximately 1 million characters.
These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make n_hidden larger.
I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that helps the validation loss.
Best models strategy
The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.

After training, we'll save the model so we can load it again later if we need too. Here I'm saving the parameters needed to create the same architecture, the hidden layer hyperparameters and the text characters.
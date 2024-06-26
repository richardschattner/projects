### Introduction
This folder showcases some experiments with ML techniques on various datasets.  
The methods used are not particularly related to another.

#### VAE
In this Notebook, VAE.ipynb , I implemented a VAE trained on the FashionMNIST dataset.
The network is able to recreate the input image quite well, albeit at the loss of some details.  
This notebook includes a short discussion on the balance of the KLD term and the reconstruction term of the VAE's loss function.
I obtained a good balance between the two through loss-terms, through experimentation.  
In particular when using the function torch.nn.functional.MSELoss, it is important to be conscious of which 'reduction' kwarg to use.
The options are either 'mean' or 'sum', which calculates the mean MSE loss and the total MSE loss of the databatch respectively.  
It is clear that $MSE_{sum}(batch) = (batch_size) \cdot MSE_{mean}(batch)$.  
In order for the batch size not to effect the MSE loss itself, it is important to choose the 'mean' reduction.  
However using only this term lead to a loss function, which was dominated by the KLD term and thus lead to a VAE, which was overly regularized and bad at reconstructing the original input, leading to a blurry 'blob'-like image.  
Through experimentation I found that multiplying the MSE_{mean} term by 32 lead to good results, while also maintining independence of the batch size.

#### Comparing data compression methods
In this notebook I compare fitting 3 NN's on the wine quality dataset.  
The first NN is fit on the data directly.  
For the second experiment, the data, consisting of 13 input features and one target variable, is first compressed into 6 input variables using PCA.
A roughly equivalent NN is then fit on this compressed data.
The final model consists of a PCA datacompression step and then a NN.  
In the third approach, an Autoencoder with 6 latent dimensions is trained first on the input data.
Then a NN is trained on the latent representation of the input data.
The final network consists of the encoder subnetwork of the trained AE and then a NN.

The results of the experiment were that fitting the NN on the original data yieled the best results, closely followed by the PCA compression model.
The AE approach achieved slightly lower performance than the first two.  
I also explored the Curse of Dimensionality, which predicst the first model to overfit more quickly than the other two (where in the third model the encoder network is frozen). 
This has been experimentally vindicated, I observed that the second model starts to overfit after $\approx 50 \% $ more training epochs than the first model.  
And in the third model, freezing the encoding network during training leads to less overfitting than when the model is allowed to also retrain the encoding layers. 

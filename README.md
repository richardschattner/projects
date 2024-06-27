## Introduction
This folder showcases some experiments with ML techniques on various datasets.  
The methods used are not particularly related to another.

### VAE (VAE.ipynb)
In this Notebook, I implemented a VAE trained on the FashionMNIST dataset.
The network is able to recreate the input image quite well, albeit at the loss of some details.  
This notebook includes a short discussion on the balance of the KLD term and the reconstruction term of the VAE's loss function.
I obtained a good balance between the two through loss-terms, through experimentation.  
In particular when using the function torch.nn.functional.MSELoss, it is important to be conscious of which 'reduction' kwarg to use.
The options are either 'mean' or 'sum', which calculates the mean MSE loss and the total MSE loss of the databatch respectively.  
It is clear that $MSE_{sum}(batch) = (batch size) \cdot MSE_{mean}(batch)$.  
In order for the batch size not to effect the MSE loss itself, it is important to choose the 'mean' reduction.  
However using only this term lead to a loss function, which was dominated by the KLD term and thus lead to a VAE, which was overly regularized and bad at reconstructing the original input, leading to a blurry 'blob'-like image.  
Through experimentation I found that multiplying the MSE_{mean} term by 32 lead to good results, while also maintining independence of the batch size.

### Comparing data compression methods (3_methods_compared.ipynb)
In this notebook I compare fitting 3 NN's on the wine quality dataset.  
The first NN is fit on the data directly.  
For the second experiment, the data, consisting of 13 input features and one target variable, is first compressed into 7 input variables using PCA.
A roughly equivalent NN is then fit on this compressed data.
The final model consists of a PCA datacompression step and then a NN.  
In the third approach, an Autoencoder with 7 latent dimensions is trained first on the input data.
Then a NN is trained on the latent representation of the input data.
The final network consists of the encoder subnetwork of the trained AE and then a NN.

The results of the experiment were that the model which used a retrainable encoder network first, achieved the best performance, followed by using PCA to compress the data first.
Fittin a NN on the dataset directly achieved a slightly higher test loss than the two.  
However, fitting a NN on the latent datarepresentation, that was learned by the AE network, without allowing the final model to change the encoders parameters, lead to significantly higher test loss.
This likely due to the relatively high reconstruction loss of the AE network on the original data, which means that the AE was unfortunately not very good at learning a lower dimensional representation of the data.  
Given this distorted and lossy compression of the input data, the relatively high test loss of the AE model, with frozen encoder network, becomes understandable.  
Note that this means, that PCA actually performed a lot better as a data compression method than the trained AE.

I also explored the Curse of Dimensionality, which predicst the first model to overfit more quickly than the other two (where in the third model the encoder network is frozen). 
This has been experimentally vindicated, I observed that the second model starts to overfit after $\approx 50 $% more training epochs than the first model.  
And in the third model, freezing the encoding network during training leads to less overfitting than when the model is allowed to also retrain the encoding layers. 

### Transformer model to predict review scores (transformer.py)
In this python program I attempt to implement a transformer model to analyze written Amazon reviews and predict the final score.
The idea is, to take reviews under a fixed length and pad shorter reviews to this length, with a special token and then feed them into the transformer, 
which uses unmasked self attention (the flash attention implementation for higher efficiency), to process the review text.

The model also sees the number of people who voted that the given review was helpful, as well as the total number of votes.  

Given that the Amazon review dataset contains relatively few tokens, when compared to normal LLM datasets, I have made the model quite small, in order to prevent overfitting and speed up learning.
In particular, my idea was, that since this task is basically a type of sentiment analysis, a small embedding space should suffice, as it is not the full meaning of words that must be learned, 
but only their effects on the review score.

This repo is currently work in progress and unfinished.
All the basic functionality has been implemented, but I have not done rigorous testing or training of the model as of yet.


#%%
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
# %%
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        outputs = self.inception(images)

        # if name, param in self.inception.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_CNN
        if isinstance(outputs, tuple) or isinstance(outputs, models.InceptionOutputs):
            features = outputs.logits
        else:
            features = outputs
            
        return self.dropout(self.ReLU(features))

'''1. Embed size = Word embeddings are dense representations of words in a lower-dimensional space, 
where words with similar meanings are close to each other
2. Vocab size = This is the number of unique words (or tokens) in your vocabulary, which determines 
the output size (the decoder needs to predict a word from the vocabulary at each step)
3. num_layers = This is the number of LSTM layers stacked together. Multiple LSTM layers can help 
the model learn more complex sequences'''    

'''The DecoderRNN takes word embeddings (created from input words), passes them through an LSTM that 
processes the sequence, and then predicts the next word at each step by mapping the LSTM's output to a probability distribution over the vocabulary.
The LSTM handles the sequence processing, while the final linear layer translates the hidden state into a word prediction.
Dropout is used to prevent overfitting during training'''
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        #nn.LSTM(embed_size, hidden_size, num_layers): The LSTM layer processes sequences of word embeddings. 
        #It takes the input (word embeddings of size embed_size) and outputs hidden states of size hidden_size for each time step in the sequence.
        #The LSTM can handle the temporal aspect of sequences, remembering context from previous words as it predicts the next word in the sequence.'''
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

#The method takes in image features and a sequence of captions.
#It embeds the captions and concatenates them with the image features to form a sequence.
#The LSTM processes this sequence, and the output hidden states are passed through a linear layer to predict the next word at each time step.
#The method returns the predicted probability distribution over the vocabulary for each word in the sequence.
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.itos[idx] for idx in result_caption]
#This method generates a caption for a given image, one word at a time.
#It starts by extracting image features using the encoder (CNN) and then iteratively predicts the next word using the decoder (RNN).
#At each step, the predicted word is fed back into the model to generate the next word in the sequence.
# The process stops when the model generates the end-of-sentence (<EOS>) token or reaches the maximum caption length.
# The final output is the sequence of words representing the caption.


    





"""
File: models.py
Author:
Description: building the model architecture for the Textract AI stack. Involves implementation of
 an CNN image encoder and a LaTeX decoder.
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models


# 2D grid positional encoding for features (output of CNN)
class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_h=50, max_w=50):
        super().__init__()
        self.d_model = d_model

        # create the positional encodings
        pos_e = torch.zeros(max_h, max_w, d_model)
        x_pos = torch.arange(0, max_w).unsqueeze(0).repeat(max_h, 1)
        y_pos = torch.arange(0, max_h).unsqueeze(1).repeat(1, max_w)

        # compute positional encoding values (using sine and cosine to create unique values in the grid)
        for i in range(0, d_model, 4):
            pos_e[:, :, i] = torch.sin(x_pos / (10000 ** (i / d_model)))
            pos_e[:, :, i+1] = torch.cos(x_pos / (10000 ** (i / d_model)))
            pos_e[:, :, i+2] = torch.sin(y_pos / (10000 ** (i / d_model)))
            pos_e[:, :, i+3] = torch.cos(y_pos / (10000 ** (i / d_model)))

        self.register_buffer('pos_e', pos_e)  # save positional encoding tensor to model

    # add the positional encoding to the features
    def forward(self, x):
        """
        :param x: features from CNN
        :return: features added with positional encoding
        """

        h, w = x.shape[2:]  # get height and width from tensor
        return x + self.pos_e[:h, :w, :].permute(2, 0, 1).unsqueeze(0)  # add positional encodings to tensor


# CNN encoder for image input to extract features
class ImageEncoder(nn.Module):

    def __init__(self, pretrained=True, frozen_stages=2, hidden_d=256):
        super().__init__()

        # we're going to use ResNet34 as a backup (base case) because it's pretrained and good in general
        rs = models.resnet34(pretrained=pretrained)

        # remove the final pooling and classification layers from it (since we're not trying to do that)
        self.backbone = nn.Sequential(*list(rs.children())[:-2])

        # in case we may want to freeze early layers
        if frozen_stages > 0:
            for i, child in enumerate(self.backbone.children()):
                if i < frozen_stages:
                    for param in child.parameters():
                        param.requires_grad = False

        # pull output channels from backbone
        dummy_inp = torch.zeros(1, 3, 224, 224)
        dummy_out = self.backbone(dummy_inp)
        backbone_channels = dummy_out.shape[1]

        # project to the hidden dimension
        self.projection = nn.Conv2d(backbone_channels, hidden_d, kernel_size=1)

        # add the positional encoding
        self.pos_encoder = PositionEncoding(hidden_d)

        # additional convolution layers (such that the output is refined)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_d, hidden_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_d, hidden_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_d),
            nn.ReLU(inplace=True)
        )

    # building forward pass
    def forward(self, images):
        """
        :param images: a given image
        :return: extracted features
        """

        # extract features, project to hidden dimension, add positional encoding, and then refine
        features = self.backbone(images)
        features = self.projection(features)
        features = self.pos_encoder(features)
        features = self.refine(features)

        return features


# using Attention to focus on specific regions of importance in the image
class Attention(nn.Module):

    def __init__(self, hidden_d, attention_d):
        super().__init__()
        self.hidden_d = hidden_d
        self.attention_d = attention_d

        # creating linear layers for attention
        self.attention_encoder = nn.Linear(hidden_d, attention_d)
        self.attention_decoder = nn.Linear(hidden_d, attention_d)
        self.full_attention = nn.Linear(attention_d, 1)

    # running forward pass
    def forward(self, encoder_out, hidden_decoder):
        """
        :param encoder_out: the encoded output from the previous layers
        :param hidden_decoder: the decoded hidden layer
        :return: the attention weights and the surrounding context
        """

        # transform encoded output
        att_1 = self.attention_encoder(encoder_out)

        # transform decoder hidden state
        att_2 = self.attention_decoder(hidden_decoder)
        att_2 = att_2.unsqueeze(1)

        # compute total attention scores
        attention = torch.tanh(att_1 + att_2)
        attention = self.full_attention(attention).squeeze(2)

        # apply softmax for weights
        sftm_weights = f.softmax(attention, dim=1)

        # compute context
        context = (encoder_out * sftm_weights.unsqueeze(2)).sum(dim=1)

        return sftm_weights, context


# LSTM class with attention to generate LaTeX expressions from the tokenized sequence
class LatexDecoder(nn.Module):

    def __init__(self, vocab_size, embed_d=256, hidden_d=256, encoder_d=256,
                 attention_d=256, num_layers=2, dropout=0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_d = hidden_d
        self.num_layers = num_layers

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_d)
        self.dropout = nn.Dropout(dropout)

        # attention
        self.attention = Attention(encoder_d, attention_d)

        # layers for lstm
        self.lstm = nn.LSTM(
            embed_d + encoder_d,
            hidden_d,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # final output projection
        self.out_proj = nn.Linear(hidden_d, vocab_size)

        # initializing decoder hidden state from the encoder output
        self.init_h = nn.Linear(encoder_d, hidden_d)
        self.init_c = nn.Linear(encoder_d, hidden_d)

    # performing initialization of decoder hidden state from the encoder output
    def init_hidden_state(self, encoder_out):
        """
        :param encoder_out: encoded output from previous layers
        :return:
        """

        # pool over dimensions
        mean_encoded_out = encoder_out.mean(dim=[2,3])

        # get hidden and context
        h = torch.tanh(self.init_h(mean_encoded_out))
        c = torch.tanh(self.init_c(mean_encoded_out))

        # expand for all LSTM layers
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return h, c

    # perform forward pass
    def forward(self, encoder_out, captions):
        """
        :param encoder_out: the encoded output from previous layers
        :param captions: targeted sequences
        :return: outputs with the attention weights
        """

        batch_size = encoder_out.size(0)
        encoder_d = encoder_out.size(1)

        # initialize our hidden state
        h, c = self.init_hidden_state(encoder_out)

        # flatten this encoded output
        encoder_out = encoder_out.view(batch_size, encoder_d, -1)
        encoder_out = encoder_out.permute(0, 2, 1)
        n_pixels = encoder_out.size(1)

        # embed captions
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)

        # initialize outputs
        max_len = captions.size(1)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)
        att_weights = torch.zeros(batch_size, max_len, n_pixels).to(encoder_out.device)

        # use actual captions as input for iterations
        for t in range(max_len):

            # get current embedding timestep
            embed_t = embeddings[:, t, :]

            # calculate attention
            weights, context = self.attention(encoder_out, h[-1])
            att_weights[:, t, :] = weights

            # synthesize weights and context
            lstm_input = torch.cat([embed_t, context], dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            # lstm forward pass
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

            # project to vocab
            outputs[:, t, :] = self.out_proj(lstm_output.squeeze(1))

        return outputs, att_weights

    # generate the LaTeX sequence using beam search
    def generate(self, encoder_out, max_len=150, start_tok=1, end_tok=2):
        """
        :param encoder_out: encoded output from previous layers
        :param max_len: maximum length for sequences
        :param start_tok: starting token
        :param end_tok: ending token
        :return: predicted sequence
        """

        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # flatten encoded output
        encoded_d = encoder_out.size(1)
        encoder_out_flat = encoder_out.view(batch_size, encoded_d, -1)
        encoder_out_flat = encoder_out_flat.permute(0, 2, 1)

        # initialize hidden state
        h, c = self.init_hidden_state(encoder_out)

        # start with SOS tokens (given these usually start any latex annotations)
        inp_tok = torch.full((batch_size,), start_tok, dtype=torch.long).to(device)
        pred_seq = []

        for _ in range(max_len):

            # embed current token we're looking at
            temp_embedding = self.embedding(inp_tok)
            temp_embedding = self.dropout(temp_embedding)

            # then compute attention
            weights, context = self.attention(encoder_out_flat, h[-1])

            # lstm input
            lstm_input = torch.cat([temp_embedding, context], dim=1).unsqueeze(1)

            # lstm forward pass
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))

            # get predictions
            output = self.out_proj(lstm_output.squeeze(1))
            pred = output.argmax(dim=1)
            pred_seq.append(pred)

            # use prediction as next input (since ya know, we're using an LSTM)
            inp_tok = pred

            # check if all sequences have produced EOS (end of sequence)
            if (pred == end_tok).all():
                break

        return torch.stack(pred_seq, dim=1)


# our image to latex class that wraps everything altogether (encoder + decoder)
class ImageToLatex(nn.Module):

    def __init__(self, vocab_size, encoder_hidden_d=256, decoder_hidden_d=256, embed_d=256,
                 attention_d=256, num_layers=2, dropout=0.5, pretrained_encoder=True):
        super().__init__()

        self.encoder = ImageEncoder(
            pretrained=pretrained_encoder,
            hidden_d=encoder_hidden_d
        )

        self.decoder = LatexDecoder(
            vocab_size=vocab_size,
            embed_d=embed_d,
            hidden_d=decoder_hidden_d,
            encoder_d=encoder_hidden_d,
            attention_d=attention_d,
            num_layers=num_layers,
            dropout=dropout
        )

    # forward pass
    def forward(self, images, captions=None):
        """
        :param images: self-explanatory
        :param captions: target sequences
        :return: outputs and attention weights (if training); predictions (if testing)
        """

        # encode the images
        encoded_out = self.encoder(images)

        if captions is not None:
            outputs, att_weights = self.decoder(encoded_out, captions)
            return outputs, att_weights
        else:
            pred = self.decoder.generate(encoded_out)
            return pred

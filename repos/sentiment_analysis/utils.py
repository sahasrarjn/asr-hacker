import re
import math
import torch
import numpy as np
import torch.nn as nn
from scipy.stats import chi
from collections import Counter
from nltk.corpus import stopwords 
from torch.nn.parameter import Parameter


def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    s = re.sub(r"\d", '', s)
    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    process = lambda x, list : list.append([onehot_dict[preprocess_string(word)] for word in x.lower().split() if preprocess_string(word) in onehot_dict.keys()])
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
        process(sent, final_list_train)

    for sent in x_val:
        process(sent, final_list_test)
            
    encoded_train = [1 if label =='positive' else 0 for label in y_train]  
    encoded_test = [1 if label =='positive' else 0 for label in y_val] 
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def quaternion_init(in_features, out_features, rng, kernel_size=None):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    s = 1. / np.sqrt(2*(fan_in + fan_out))
    rng = np.random.RandomState(np.random.randint(1,1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (torch.from_numpy(weight_r), torch.from_numpy(weight_i), torch.from_numpy(weight_j), torch.from_numpy(weight_k))

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=None):
        super(QLinear, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale_param  = None
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = np.random.RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = quaternion_init
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

        kernel_size = None
        r, i, j, k  = winit(self.r_weight.size(0), self.r_weight.size(1), self.rng, kernel_size)
        self.r_weight.data = r.type_as(self.r_weight.data)
        self.i_weight.data = i.type_as(self.i_weight.data)
        self.j_weight.data = j.type_as(self.j_weight.data)
        self.k_weight.data = k.type_as(self.k_weight.data)

    def forward(self, input):
        kernels_4_r = torch.cat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], dim=0)
        kernels_4_i = torch.cat([self.i_weight,  self.r_weight, -self.k_weight, self.j_weight], dim=0)
        kernels_4_j = torch.cat([self.j_weight,  self.k_weight, self.r_weight, -self.i_weight], dim=0)
        kernels_4_k = torch.cat([self.k_weight,  -self.j_weight, self.i_weight, self.r_weight], dim=0)
        kernels_4_quaternion   = torch.cat([kernels_4_r, kernels_4_i, kernels_4_j, kernels_4_k], dim=1)

        if input.dim() == 2 :

            if self.bias is not None:
                return torch.addmm(self.bias, input, kernels_4_quaternion)
            else:
                return torch.mm(input, kernels_4_quaternion)
        else:
            output = torch.matmul(input, kernels_4_quaternion)
            if self.bias is not None:
                return output+self.bias
            else:
                return output


class CustomLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, quaternion=False):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        if quaternion: linear_layer = QLinear
        else: linear_layer = nn.Linear
        
        #i_t
        self.W_i = linear_layer(input_sz, hidden_sz)
        self.U_i = linear_layer(hidden_sz, hidden_sz, bias=False)
        
        #f_t
        self.W_f = linear_layer(input_sz, hidden_sz)
        self.U_f = linear_layer(hidden_sz, hidden_sz, bias=False)
        
        #c_t
        self.W_c = linear_layer(input_sz, hidden_sz)
        self.U_c = linear_layer(hidden_sz, hidden_sz, bias=False)
        
        #o_t
        self.W_o = linear_layer(input_sz, hidden_sz)
        self.U_o = linear_layer(hidden_sz, hidden_sz, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, init_states=None):
        """
        x.shape == (batch_size, sequence_size, input_size)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size),
                torch.zeros(bs, self.hidden_size),
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_t))
            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_t))
            g_t = torch.tanh(self.W_c(x_t) + self.U_c(h_t))
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_t))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)    
            hidden_seq.append(h_t.unsqueeze(0))
        
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
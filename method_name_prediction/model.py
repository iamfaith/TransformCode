from aicmder.torch.transformer import TransformerEncoder, TransformerDecoder, Embeddings, PADDING_INDEX, ATTEN_TYPE_SELF, _position_encoding_default, _position_encoding_absolute, DotProductAttention, ATTEN_TYPE_HYGRA
from aicmder.torch.transformer import ActivationFunction, ACTIVATION_FUNCTIONS
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from aicmder.torch import Module
import torch
import torch.nn as nn
import numpy as np


import logging
import json

logger = logging.getLogger(__name__)



class Encoder(Module):
    def __init__(self,
                 #  wocab_size,
                 #  atten_dim,
                 config,
                 d_model=512,
                 num_layers=2,
                 heads=8,
                 dropout=0.2,
                 max_relative_positions=32,
                 d_ff=2048,
                 attn_type=ATTEN_TYPE_SELF,
                 position_encoding=_position_encoding_default,
                 activation_fn=ActivationFunction.log_softmax):
        super().__init__()
        self.save_hyperparameters()

        self.config.hidden_size = d_model
        self.embeddings = RobertaEmbeddings(self.config)
        # self.word_embeddings = Embeddings(d_model, wocab_size, PADDING_INDEX, position_encoding=position_encoding)

        ############################################################################## not work
        # # # 创建一个TransformerEncoderLayer实例
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dropout=dropout, dim_feedforward=d_ff)
        # # # 创建一个TransformerEncoder实例
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.transformer_encoder = TransformerEncoder(
            heads, d_model, dropout, max_relative_positions, d_ff, embeddings=None,
            num_layers=num_layers, attn_type=attn_type, check_device=False)
    
        prev_dim = d_model
        dim = d_model
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True),  # first layer
                                nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True),  # second layer
                                nn.Linear(prev_dim, dim, bias=False),
                                nn.BatchNorm1d(dim, affine=False))  # output layer
        pred_dim = 128
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer
        # with open("/mnt/source-code-efficient-ft/OJ_idf.json", "r") as fp:
            # self.idf_dict = json.load(fp)

    def forward(
            self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None,
            past_key_values=None, output_attentions=False, labels=None, return_mean=True, mask=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        if isinstance(self.transformer_encoder, nn.TransformerEncoder):
            out = self.transformer_encoder(embedding_output)
            vec = out.mean(dim=1)
        else: 
            out = self.transformer_encoder(embedding_output, output_attentions=output_attentions)

            ####################################  add idf
            if hasattr(self, "idf_dict"):
                nb, size = input_ids.shape
                idf_weight = []
                for i in range(nb):
                    temp = []
                    for j in input_ids[i]:
                        temp.append(self.idf_dict[str(j.item())])
                    idf_weight.append(temp)
                idf_weight = torch.Tensor(idf_weight).to(device)

                vec = torch.mul(out[0], idf_weight.unsqueeze(-1))
                vec = vec.mean(dim=1)
            ####################################  add idf

            ######################### origin
            else:
                vec = out[0]
                if mask is not None:
                    attention_mask = mask.unsqueeze(-1).expand(out[0].size()).float()
                    vec = vec * attention_mask
                if return_mean:
                    vec = vec.mean(dim=1)


        ################################ add projector
        if hasattr(self, "fc") and return_mean:
            vec = self.predictor(self.fc(vec))

        if output_attentions:
            return vec, out[1]
        # 0 -- hidden state  1 -- attns

        if labels is not None:
            # lm_logits = lm_logits.to(torch.float32)
            # Shift so that tokens < n predict n
            shift_logits = vec[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # lm_logits = lm_logits.to(hidden_states.dtype)
            # loss = loss.to(hidden_states.dtype)
            return vec, loss
        return vec


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class CodeEmbed(Module):
    def __init__(self,
                 #  wocab_size,
                 #  atten_dim,
                 config,
                 encoder=None,
                 tokenizer=None,
                 K=96000,
                #  K=43000,
                #  K=65536,/ 
                 m=0.999, T=0.07,
                 d_model=128,
                 num_layers=2,
                 heads=8,
                 dropout=0.2,
                 label_size=2,
                 max_relative_positions=32,
                 d_ff=2048,
                 attn_type=ATTEN_TYPE_SELF,
                 position_encoding=_position_encoding_default,
                 activation_fn=ActivationFunction.log_softmax):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.save_hyperparameters()

        print("model config:", d_model, num_layers, heads, d_ff)
        config.hidden_size = d_model
        
        if encoder is None:
            self.encoder_q = Encoder(config, d_model=d_model, num_layers=num_layers, heads=heads, dropout=dropout, max_relative_positions=max_relative_positions, d_ff=d_ff, position_encoding=position_encoding, activation_fn=activation_fn)
            self.encoder_k = Encoder(config, d_model=d_model, num_layers=num_layers, heads=heads, dropout=dropout, max_relative_positions=max_relative_positions, d_ff=d_ff, position_encoding=position_encoding, activation_fn=activation_fn)
            self.tokenizer = None
        else:
            self.encoder_q = encoder
            self.encoder_k = encoder
            self.tokenizer = tokenizer


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(d_model, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.flatten = nn.Flatten()
        self.hidden2label = nn.Linear(d_model * label_size, label_size)

        # self.register_parameter(name='alpha', param=nn.Parameter(torch.tensor([0.2])))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, "K: {} bs: {}".format(self.K, batch_size)  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    
    @torch.no_grad()
    def encode(self, x1, x2=None):
        vec2 = None
        if self.tokenizer is not None:
            attention_mask_x1 = x1.ne(self.tokenizer.pad_token_id)
            hidden_states = self.encoder_q(input_ids=x1, attention_mask=attention_mask_x1)[0]
            vec1 = hidden_states.mean(dim=1)
            if x2 is not None:
                attention_mask_x2 = x2.ne(self.tokenizer.pad_token_id)
                hidden_states = self.encoder_q(input_ids=x2, attention_mask=attention_mask_x2)[0]
                vec2 = hidden_states.mean(dim=1)        
        else:
            vec1 = self.encoder_q(x1)
            if x2 is not None:
                vec2 = self.encoder_q(x2)
        # return vec1, vec2
        tree = torch.cat((vec1, vec2), dim=1)
        y = self.hidden2label(self.flatten(tree))
        log_probs = ACTIVATION_FUNCTIONS[self.activation_fn](y)
        return log_probs

    @torch.no_grad()
    def represent(self, x, output_attentions=True):
        return self.encoder_q(x, output_attentions=output_attentions)

    def forward(self, x1, x2=None):
        """
        Input:
            im_q: a batch of query images (x1)
            im_k: a batch of key images (x2)
        Output:
            logits, targets
        """
        if not self.training:
            return self.encode(x1, x2)

        # compute query features
        if self.tokenizer is not None:
            attention_mask_x1 = x1.ne(self.tokenizer.pad_token_id)
            hidden_states = self.encoder_q(input_ids=x1, attention_mask=attention_mask_x1)[0]
            q = hidden_states.mean(dim=1)
        else:
            q = self.encoder_q(x1)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            if self.tokenizer is not None:
                attention_mask_x2 = x2.ne(self.tokenizer.pad_token_id)
                hidden_states = self.encoder_k(input_ids=x2, attention_mask=attention_mask_x2)[0]
                k = hidden_states.mean(dim=1)
            else:
                k = self.encoder_k(x2)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)


        tree = torch.cat((q, self.encoder_q(x2)), dim=1)
        # print(tree.shape)
        y = self.hidden2label(self.flatten(tree))
        log_probs = ACTIVATION_FUNCTIONS[self.activation_fn](y)
        # print("---", tree.shape, log_probs.shape)
        return logits, labels, log_probs



from commode_utils.losses import SequenceCrossEntropyLoss
# from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
# from commode_utils.metrics.chrF import ChrF
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
class MethodNamePredictor(Module):
    def __init__(self,
                 #  wocab_size,
                 #  atten_dim,
                 config,
                 vocab_size=200,
                 bos_id=1,
                 blank_id=0,
                 encoder=None,
                 tokenizer=None,
                 d_model=256,
                 num_layers=6,
                 heads=8,
                 dropout=0.2,
                 max_relative_positions=32,
                 d_ff=2048,
                 attn_type=ATTEN_TYPE_SELF,
                 position_encoding=_position_encoding_default,
                 activation_fn=ActivationFunction.log_softmax):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.save_hyperparameters()

        print("model config:", d_model, num_layers, heads, d_ff)
        config.hidden_size = d_model
        
        if encoder is None:
            self.encoder_q = Encoder(config, d_model=d_model, num_layers=num_layers, heads=heads, dropout=dropout, max_relative_positions=max_relative_positions, d_ff=d_ff, position_encoding=position_encoding, activation_fn=activation_fn)
            self.tokenizer = None
        else:
            self.encoder_q = encoder
            self.tokenizer = tokenizer


        # self.llm_model = llm_model
        # get the hidden size of the LLM model
        hidden_size = config.hidden_size
       
        self.__pad_idx = self.blank_id
        self.__sos_id = 1 # vocabulary.label_to_id[vocabulary.SOS]

        model_config = DictConfig({
            'embedding_size': hidden_size,
            'decoder_num_layers': 1,
            'decoder_size': hidden_size,
            'rnn_dropout': 0.5
        })
        teacher_forcing = 1.0
        decoder_step = LSTMDecoderStep(model_config, vocab_size, self.__pad_idx)
        self._decoder = Decoder(
            decoder_step, vocab_size, self.__sos_id, teacher_forcing
        )

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")

    def forward(self, input_ids, mask=None, labels=None, contexts_per_label=None, output_length=7):
        
        # get the batch size and input length
        batch_size = contexts_per_label.size(0)
        # input_ids = input_ids.transpose(0, 1)
        # get the LLM output
        # output: [batch_size, input_length, hidden_size]
        # output = self.encoder_q(input_ids).last_hidden_state
        output = self.encoder_q(input_ids, return_mean=True, mask=mask)
        encoded_paths = output #.reshape(-1, output.shape[0])
        if labels is not None:
            labels = labels.transpose(0, 1)
            output_length = labels.size(0)
        target_sequence = labels if self.training else None
        output_logits, attention_weights = self._decoder(
            encoded_paths, contexts_per_label, output_length, target_sequence
        )
        if self.training:
            return self.__loss(output_logits[1:], labels[1:]), output_logits
        else:
            return output_logits

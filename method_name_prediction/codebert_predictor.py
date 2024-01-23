from aicmder.torch import Module
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)


class MethodNamePredictor(Module):
    def __init__(self,
                 config,
                 vocab_size=200,
                 bos_id=1,
                 blank_id=0,
                 tokenizer=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        self.save_hyperparameters()



        config_class, model_class  = RobertaConfig, RobertaModel
        self.config = config_class.from_pretrained('microsoft/codebert-base', output_attentions=True)


        self.model = model_class.from_pretrained('microsoft/codebert-base', config=config)
        self.tokenizer = tokenizer #tokenizer_class.from_pretrained('roberta-base')

        for param in self.model.parameters():
            param.requires_grad = False

        hidden_size = self.config.hidden_size

        self.__pad_idx = self.blank_id
        self.__sos_id = 1  # vocabulary.label_to_id[vocabulary.SOS]

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

    def forward(self, input_ids, labels=None, contexts_per_label=None, output_length=7):

        output = self.model(input_ids) 
        encoded_paths = output[0].mean(dim=1)  
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

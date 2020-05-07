from typing import Any, Dict, List, Optional
import logging
from collections import OrderedDict

import torch
from torch.nn import CrossEntropyLoss

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from pytorch_transformers import BertModel

logger = logging.getLogger(__name__)


@Model.register("nq")
class BertForNaturalQuestionAnswering(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_pretrained_model: str,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)
        self.bert_model = BertModel.from_pretrained(bert_pretrained_model)
        bert_dim = self.bert_model.pooler.dense.out_features
        self.qa_outputs = torch.nn.Linear(bert_dim, 2)
        initializer(self)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions=None,
                end_positions=None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration
                (:class:`~transformers.BertConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
                Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-start scores (before SoftMax).
            end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
                Span-end scores (before SoftMax).
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :param start_positions: Start position if it's training data. None for testing.
        :param end_positions: End position if it's training data. None for testing.
        :return:
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        output = {
            'start_logits': start_logits,
            'end_logits': end_logits
        }
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output['loss'] = total_loss

        return output  # (loss), start_logits, end_logits, (hidden_states), (attentions)

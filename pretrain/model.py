import math
from itertools import chain
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _softmax_backward_data as _softmax_backward_data
from torch.utils import checkpoint


class Bert(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.embedding = Embedding(config)
        self.transformer = Encoder(config, activation_checkpointing)

        self.teacher_embedding = copy.deepcopy(self.embedding)
        self.teacher = copy.deepcopy(self.transformer)
        for param in chain(self.teacher.parameters(), self.teacher_embedding.parameters()):
            param.requires_grad = False

        decoder_config = copy.deepcopy(config)
        decoder_config.num_hidden_layers = 4
        decoder_config.attention_probs_dropout_prob = 0.0
        decoder_config.hidden_dropout_prob = 0.0
        self.decoder = Encoder(decoder_config, activation_checkpointing)

        self.classifier = MaskClassifier(config, self.embedding.word_embedding.weight)
        self.byol_classifier = ByolClassifier(config)

    def get_contextualized(self, encoder_inputs, outputs, unmasked_ids, encoder_attention_mask, context_attention_mask):
        static_embeddings, relative_embedding, mask_embedding = self.embedding(encoder_inputs)
        encoded_embeddings = self.transformer(static_embeddings, encoder_attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding, unmasked_ids)

        # remap
        encoder_output = torch.scatter_add(
            input=torch.zeros(context_attention_mask.size(1), context_attention_mask.size(0), encoded_embeddings.size(-1), device=encoded_embeddings.device),
            dim=0,
            index=unmasked_ids.unsqueeze(-1).expand(-1, -1, encoded_embeddings.size(-1)),
            src=encoded_embeddings.masked_fill(encoder_attention_mask.t().unsqueeze(-1), 0.0)
        )
        decoder_input = torch.where(outputs.eq(-100).unsqueeze(-1), encoder_output, mask_embedding)

        decoder_output = self.decoder(decoder_input, context_attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding)
        decoder_output = torch.index_select(
            decoder_output.flatten(0, 1), 0, torch.nonzero(outputs.flatten() != -100).squeeze()
        )
        return decoder_output

    @torch.no_grad()
    def get_teacher_embedding(self, input_ids, attention_mask, masked_lm_labels):
        self.teacher_embedding = self.teacher_embedding.eval()
        self.teacher = self.teacher.eval()

        static_embeddings, relative_embedding, _ = self.teacher_embedding(input_ids)
        hidden_states = self.teacher(static_embeddings, attention_mask.unsqueeze(1).unsqueeze(2), relative_embedding)
        hidden_states = torch.index_select(
            hidden_states.flatten(0, 1), 0, torch.nonzero(masked_lm_labels.flatten() != -100).squeeze()
        )

        hidden_states = F.layer_norm(hidden_states, hidden_states.shape[-1:])
        return hidden_states

    def forward(self, encoder_inputs, full_inputs, outputs, unmasked_ids, encoder_attention_mask, context_attention_mask):
        contextualized_embeddings = self.get_contextualized(encoder_inputs, outputs, unmasked_ids, encoder_attention_mask, context_attention_mask)
        teacher_embeddings = self.get_teacher_embedding(full_inputs, context_attention_mask, outputs)

        subword_prediction = self.classifier(contextualized_embeddings)
        byol_prediction = self.byol_classifier(contextualized_embeddings)

        flatten_labels = torch.flatten(outputs)
        flatten_labels = flatten_labels[flatten_labels != -100]
        mlm_loss = F.cross_entropy(subword_prediction, flatten_labels)
        byol_loss = F.smooth_l1_loss(byol_prediction, teacher_embeddings)

        with torch.no_grad():
            accuracy = (subword_prediction.argmax(-1) == flatten_labels).float().mean()

        return mlm_loss, byol_loss, accuracy


class Encoder(nn.Module):
    def __init__(self, config, activation_checkpointing=False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)

        for i, layer in enumerate(self.layers):
            layer.mlp.mlp[-2].weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))
            layer.attention.post_layer_norm.weight.data *= math.sqrt(1.0 / (2.0 * (1 + i)))

        self.activation_checkpointing = activation_checkpointing

    def forward(self, hidden_states, attention_mask, relative_embedding, applied_position_ids=None):
        for layer in self.layers:
            if self.activation_checkpointing:
                hidden_states = checkpoint.checkpoint(layer, hidden_states, attention_mask, relative_embedding, applied_position_ids)[0]
            else:
                hidden_states = layer(hidden_states, attention_mask, relative_embedding, applied_position_ids)[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states


class MaskClassifier(nn.Module):
    def __init__(self, config, subword_embedding):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(subword_embedding.size(1), subword_embedding.size(0))
        )
        self.initialize(config.hidden_size, subword_embedding)

    def initialize(self, hidden_size, embedding):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[-1].weight = embedding
        self.nonlinearity[0].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x


class ByolClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nonlinearity = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.nonlinearity[0].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.nonlinearity[-1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.nonlinearity[0].bias.data.zero_()
        self.nonlinearity[-1].bias.data.zero_()

    def forward(self, x):
        x = self.nonlinearity(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, padding_mask, relative_embedding, applied_position_ids=None):
        x_, attention_scores = self.attention(x, padding_mask, relative_embedding, applied_position_ids)
        x = x + x_
        x_ = self.mlp(x)
        x = x + x_
        return x, attention_scores, x_


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, 2*config.intermediate_size, bias=False),
            GeGLU(),
            nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.initialize(config.hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.mlp[1].weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.mlp[-2].weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return self.mlp(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(self, x, mask, dim):
        self.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, self.dim)
        x.masked_fill_(mask, 0.0)
        self.save_for_backward(x)
        return x

    @staticmethod
    def backward(self, grad_output):
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output.dtype)
        return inputGrad, None, None


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads

        self.in_proj_qk = nn.Linear(config.hidden_size, 2*config.hidden_size, bias=True)
        self.in_proj_v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.pre_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=False)
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        position_indices = torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(1) \
            - torch.arange(config.max_position_embeddings, dtype=torch.long).unsqueeze(0)
        position_indices = self.make_log_bucket_position(position_indices, config.position_bucket_size, config.max_position_embeddings)
        position_indices = config.position_bucket_size - 1 + position_indices
        self.register_buffer("position_indices", position_indices, persistent=True)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = 1.0 / math.sqrt(3 * self.head_size)
        self.initialize()

    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        sign = torch.sign(relative_pos)
        mid = bucket_size // 2
        abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos).clamp(max=max_position - 1))
        log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position-1) / mid) * (mid - 1)).int() + mid
        bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign).long()
        return bucket_pos

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.in_proj_qk.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.in_proj_v.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.in_proj_qk.bias.data.zero_()
        self.in_proj_v.bias.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(self, hidden_states, attention_mask, relative_embedding, applied_position_ids=None):
        key_len, batch_size, _ = hidden_states.size()
        query_len = key_len

        if self.position_indices.size(0) < query_len:
            position_indices = torch.arange(query_len, dtype=torch.long).unsqueeze(1) \
                - torch.arange(query_len, dtype=torch.long).unsqueeze(0)
            position_indices = self.make_log_bucket_position(position_indices, self.config.position_bucket_size, 256)
            position_indices = self.config.position_bucket_size - 1 + position_indices
            self.register_buffer("position_indices", position_indices.to(hidden_states.device), persistent=True)

        hidden_states = self.pre_layer_norm(hidden_states)

        query, key = self.in_proj_qk(hidden_states).chunk(2, dim=2)  # shape: [T, B, D]
        value = self.in_proj_v(hidden_states)  # shape: [T, B, D]

        query_pos, key_pos = self.in_proj_qk(self.dropout(relative_embedding)).chunk(2, dim=-1)  # shape: [2T-1, D]
        query_pos = query_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]
        key_pos = key_pos.view(-1, self.num_heads, self.head_size)  # shape: [2T-1, H, D]

        query = query.reshape(query_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        key = key.reshape(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)
        value = value.view(key_len, batch_size * self.num_heads, self.head_size).transpose(0, 1)

        attention_scores = torch.bmm(query, key.transpose(1, 2) * self.scale)

        query = query.view(batch_size, self.num_heads, query_len, self.head_size)
        key = key.view(batch_size, self.num_heads, query_len, self.head_size)
        attention_scores = attention_scores.view(batch_size, self.num_heads, query_len, key_len)
        
        attention_scores_qp = torch.einsum("bhqd,khd->bhqk", query, key_pos * self.scale)  # shape: [B, H, Tq, Tr]
        attention_scores_pk = torch.einsum("bhkd,qhd->bhqk", key * self.scale, query_pos)  # shape: [B, H, Tr, Tk]
        if applied_position_ids is not None:
            position_indices = self.position_indices.expand(batch_size, -1, -1)  # shape: [B, T, T]
            position_indices = position_indices.gather(1, applied_position_ids.t().unsqueeze(-1).expand(-1, -1, position_indices.size(-1)))  # shape: [B, T, T]
            position_indices = position_indices.gather(2, applied_position_ids.t().unsqueeze(1).expand(-1, position_indices.size(1), -1))  # shape: [B, T, T]
            position_indices = position_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # shape: [B, H, T, T]
        else:
            position_indices = self.position_indices[:query_len, :key_len].expand(batch_size, self.num_heads, -1, -1)
        attention_scores_qp = attention_scores_qp.gather(dim=-1, index=position_indices)  # shape: [B, H, Tq, Tk]
        attention_scores_pk = attention_scores_pk.gather(dim=-2, index=position_indices)  # shape: [B, H, Tq, Tk]
        attention_scores.add_(attention_scores_qp)
        attention_scores.add_(attention_scores_pk)

        returned_attention_scores = attention_scores.detach().clone().mean(dim=1)

        attention_probs = MaskedSoftmax.apply(attention_scores, attention_mask, -1)

        attention_probs = self.dropout(attention_probs)
        context = torch.bmm(attention_probs.flatten(0, 1), value)  # shape: [B*H, Q, D]
        context = context.transpose(0, 1).reshape(context.size(1), -1, self.hidden_size)  # shape: [Q, B, H*D]
        context = self.out_proj(context)
        context = self.post_layer_norm(context)
        context = self.dropout(context)

        return context, returned_attention_scores


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.mask_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.relative_embedding = nn.Parameter(torch.empty(2 * config.position_bucket_size - 1, config.hidden_size))
        self.relative_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.initialize()

    def initialize(self):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))
        nn.init.trunc_normal_(self.relative_embedding, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids):
        word_embedding = self.dropout(self.word_layer_norm(self.word_embedding(input_ids)))
        relative_embeddings = self.relative_layer_norm(self.relative_embedding)
        return word_embedding, relative_embeddings, self.mask_embedding

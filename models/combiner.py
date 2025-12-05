import torch
from torch import nn
import torch.nn.functional as F
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from collections import OrderedDict
import copy

class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def predict_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        # output = self.output_layer(combined_features) + text_features + image_features
        return output

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        # output = self.output_layer(combined_features) + text_features + image_features
        return F.normalize(output, dim=-1)


class myCrossAttn(nn.Module):
    def __init__(self, num_layers,d_model):
        super().__init__()
        self.layers = _get_clones(TransformerDecoderLayer(d_model,8), num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)


    def forward(self, query, memory):
        output = query
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory)

        if self.norm is not None:
            output = self.norm(output)

        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

from typing import Optional, List
from torch import Tensor
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



                # class CrossFormer(nn.Module):
#     """
#     Combiner module which once trained fuses textual and visual information
#     """
#     def __init__(self, embed_dim, cmt_depth, vocab_size):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.vocab_size = vocab_size
#         self.cross_attn = nn.MultiheadAttention(self.embed_dim,
#                                                 self.embed_dim // 64,
#                                                 batch_first=True)
#         self.cross_modal_transformer = Transformer(width=self.embed_dim,
#                                                    layers=cmt_depth,
#                                                    heads=self.embed_dim //
#                                                          64)
#         scale = self.cross_modal_transformer.width ** -0.5
#
#         self.ln_pre_t = LayerNorm(self.embed_dim)
#         self.ln_pre_i = LayerNorm(self.embed_dim)
#         self.ln_post = LayerNorm(self.embed_dim)
#
#         proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
#         attn_std = scale
#         fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
#         for block in self.cross_modal_transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#
#         # init cross attn
#         nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
#         nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
#
#         self.mlm_head = nn.Sequential(
#             OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
#                          ('gelu', QuickGELU()),
#                          ('ln', LayerNorm(self.embed_dim)),
#                          ('fc', nn.Linear(self.embed_dim, vocab_size))]))
#         # init mlm head
#         nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
#         nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
#
#         self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)
#
#     def cross_former(self, q, k, v):
#         x = self.cross_attn(
#             self.ln_pre_t(q),
#             self.ln_pre_i(k),
#             self.ln_pre_i(v),
#             need_weights=False)[0]
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.cross_modal_transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_post(x)
#         return x
#
#     def forward(self, text_features: torch.tensor, image_features: torch.tensor,
#                 mlm_labels: torch.tensor) -> torch.tensor:
#         image_features = image_features.permute(1, 0, 2)  # LND -> NLD
#         mlm_labels = mlm_labels.reshape(-1)
#         x = self.cross_former(text_features, image_features, image_features)
#         # print(text_features.shape, image_features.shape, mlm_labels.shape)
#         x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
#         # print(x.shape)
#         scores = x.float().reshape(-1, self.vocab_size)
#         # print(scores.shape)
#         return self.mlm_loss(scores, mlm_labels)


# class Combiner(nn.Module):
#     """
#     Combiner module which once trained fuses textual and visual information
#     """
#     def __init__(self, embed_dim, cmt_depth, vocab_size):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.vocab_size = vocab_size
#         self.cross_attn = nn.MultiheadAttention(self.embed_dim,
#                                                 self.embed_dim // 64,
#                                                 batch_first=True)
#         self.cross_modal_transformer = Transformer(width=self.embed_dim,
#                                                    layers=cmt_depth,
#                                                    heads=self.embed_dim //
#                                                          64)
#         scale = self.cross_modal_transformer.width ** -0.5
#
#         self.ln_pre_t = LayerNorm(self.embed_dim)
#         self.ln_pre_i = LayerNorm(self.embed_dim)
#         self.ln_post = LayerNorm(self.embed_dim)
#
#         proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
#         attn_std = scale
#         fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
#         for block in self.cross_modal_transformer.resblocks:
#             nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#             nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#             nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#             nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#
#         # init cross attn
#         nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
#         nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
#
#         self.mlm_head = nn.Sequential(
#             OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
#                          ('gelu', QuickGELU()),
#                          ('ln', LayerNorm(self.embed_dim)),
#                          ('fc', nn.Linear(self.embed_dim, vocab_size))]))
#         # init mlm head
#         nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
#         nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
#
#         self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)
#
#         clip_feature_dim = 1024
#         projection_dim = 1024
#         hidden_dim = 2048
#         self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
#         self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
#
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.5)
#
#         self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
#
#         self.dropout3 = nn.Dropout(0.5)
#         self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
#                                             nn.Linear(hidden_dim, 1), nn.Sigmoid())
#
#     def cross_former_tii(self, q, k, v):
#         x = self.cross_attn(
#             self.ln_pre_t(q),
#             self.ln_pre_i(k),
#             self.ln_pre_i(v),
#             need_weights=False)[0]
#         # x = x.permute(1, 0, 2)  # NLD -> LND
#         # x = self.cross_modal_transformer(x)
#         # x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_post(x)
#         return x
#
#     def cross_former_itt(self, q, k, v):
#         x = self.cross_attn(
#             self.ln_pre_i(q),
#             self.ln_pre_t(k),
#             self.ln_pre_t(v),
#             need_weights=False)[0]
#         # x = x.permute(1, 0, 2)  # NLD -> LND
#         # x = self.cross_modal_transformer(x)
#         # x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_post(x)
#         return x
#
#     def forward(self, text: torch.tensor, text_features: torch.tensor, image_features: torch.tensor) -> torch.tensor:
#         # image_features = image_features.permute(1, 0, 2)  # LND -> NLD
#         # mlm_labels = mlm_labels.reshape(-1)
#
#         t = self.cross_former_tii(text_features, image_features, image_features)
#         # x = x.permute(1, 0, 2)  # NLD -> LND
#         # # print(text_features.shape, image_features.shape, mlm_labels.shape)
#         # x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
#         t = t[torch.arange(t.shape[0]), text.argmax(dim=-1)]
#         # t = t[:, 0, :]
#
#
#         # image_features = image_features.permute(1, 0, 2)  # LND -> NLD
#         # text_features = text_features.permute(1, 0, 2)  # LND -> NLD
#         # i = self.cross_former_itt(image_features, text_features, text_features)
#         # i = i[:, 0, :]
#
#
#         # scores = x.float().reshape(-1, self.vocab_size)
#         # # print(scores.shape)
#         # return self.mlm_loss(scores, mlm_labels)
#         return t
#
#     def predict_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
#         """
#         Combine the reference image features and the caption features. It outputs the predicted features
#         :param image_features: CLIP reference image features
#         :param text_features: CLIP relative caption features
#         :return: predicted features
#         """
#         t = self.cross_former_tii(text_features, image_features, image_features)
#         t = t[torch.arange(t.shape[0]), text.argmax(dim=-1)]
#
#         text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
#         image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
#
#         raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
#         # combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
#         dynamic_scalar = self.dynamic_scalar(raw_combined_features)
#         output = t + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
#         # output = self.output_layer(combined_features)
#         return output
#
#     def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
#         """
#         Combine the reference image features and the caption features. It outputs the predicted features
#         :param image_features: CLIP reference image features
#         :param text_features: CLIP relative caption features
#         :return: predicted features
#         """
#         t = self.cross_former_tii(text_features, image_features, image_features)
#         t = t[torch.arange(t.shape[0]), text.argmax(dim=-1)]
#         text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
#         image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
#
#         raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
#         combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
#         dynamic_scalar = self.dynamic_scalar(raw_combined_features)
#         # output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
#         #         1 - dynamic_scalar) * image_features
#         output = t + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
#         # output = self.output_layer(combined_features)
#         return F.normalize(output, dim=-1)



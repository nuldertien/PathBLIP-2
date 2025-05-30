"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime
import re
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.identity_vision_encoder import create_identity_model
from lavis.models.adaptive_pooling import create_adaptive_pooling
from lavis.models.padding_per_batch import create_padding_per_batch
from transformers import BertTokenizer, AutoTokenizer


class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    # def init_tokenizer(cls, truncation_side="right"):
    #     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
    #     tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    #     return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        print("VISION WIDTH")
        print(vision_width)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, 
        precision, technique, max_patches
    ):
        vision_agg_models = [
            "identity",
            "adaptive_pooling",
            "padding_per_batch",
        ]
        assert model_name in vision_agg_models, \
            f"vit / aggregation model must be {vision_agg_models}"
        if model_name == "identity":
            visual_encoder = create_identity_model()
        elif model_name == "adaptive_pooling":
            visual_encoder = create_adaptive_pooling(
                max_patches=max_patches,
                technique=technique
            )
        elif model_name == "padding_per_batch":
            visual_encoder = create_padding_per_batch(
                max_patches=max_patches
            )
        ln_vision = LayerNorm(visual_encoder.num_features) 
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        import json
        "Param groups = %s" % json.dumps(parameter_group_names, indent=2)
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    def custom_truncate(self, full_text, he_text, max_len, tokenizer):
        """
        Truncate text to max_len
        """
        tokenized = tokenizer(
            full_text, 
            return_tensors='pt', 
            padding=True
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Check if truncation is needed
        if input_ids.size(1) <= max_len:
            return full_text

        # Truncate text
        for idx, mask in enumerate(attention_mask):
            if mask.sum() > max_len:
                full_text[idx] = self.randomized_truncate(
                    full_text[idx],
                    he_text[idx], 
                    max_len, 
                    tokenizer)
        
        return full_text

    def randomized_truncate(self, full_single_text, he_single_text, 
                            max_len, tokenizer):
        """
        Randomly truncate text to max_len.
        """
        sentence_end_pattern = r'(?<=[.?!])(?=\s(?![0-9]))|(?<=\n)'
        # Split full text into sentences
        full_sentences = {
            idx: sentence.strip()
            for idx, sentence in enumerate(re.split(
                sentence_end_pattern, full_single_text
                ))
            if sentence.strip()
        }
        
        he_sentences = {
            idx: sentence.strip()
            for idx, sentence in enumerate(re.split(
                sentence_end_pattern, he_single_text
                ))
            if sentence.strip()
        }
        
        # Match sentences
        new_text, filtered_full_sentences = self.match_sentences_to_text(
            full_sentences.copy(), he_sentences.copy(), n_words=2
        )
        
        # Pre-tokenize sentences for efficiency
        sentence_tokens = {
            idx: len(tokenizer(sentence, 
                            return_tensors='pt', 
                            truncation=False,
                            add_special_tokens=False).input_ids[0])
            for idx, sentence in filtered_full_sentences.items()
        }
        
        total_token_length = sum(
            len(tokenizer(sentence, return_tensors='pt', truncation=False).input_ids[0])
            for sentence in new_text.values()
        )
        
        # Randomly add sentences until max_len is reached
        keys_to_add = list(filtered_full_sentences.keys())
        while total_token_length < max_len and keys_to_add:
            random_key = random.choice(keys_to_add)
            random_sentence = filtered_full_sentences.pop(random_key)
            keys_to_add.remove(random_key)
            
            sentence_length = sentence_tokens[random_key]
            if total_token_length + sentence_length > max_len:
                break
            new_text[random_key] = random_sentence
            total_token_length += sentence_length
        
        # Sort new_text by keys
        ordered_new_text = OrderedDict(sorted(new_text.items()))
        text_joined = " ".join(ordered_new_text.values())
        
        return text_joined

    def match_sentences_to_text(self, full_sentences, he_sentences, n_words=None):
        # Truncate he_sentences if n_words is specified
        if n_words:
            he_sentences = {
                idx: " ".join(sentence.split()[:n_words])
                for idx, sentence in he_sentences.items()
            }
        
        new_text = {}
        for he_idx, he_sentence in he_sentences.items():
            match_found = False
            for full_idx, full_sentence in full_sentences.items():
                if he_sentence.lower() in full_sentence.lower():
                    new_text[full_idx] = full_sentence
                    match_found = True
                    break
            if not match_found:
                pass
                # logging.info(f"No match found for: {he_sentence}")
        
        # Remove matched sentences
        matched_full_indices = set(new_text.keys())
        full_sentences = {idx: sent for idx, sent in full_sentences.items() if idx not in matched_full_indices}
        
        return new_text, full_sentences

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

def pathology_compute_sim_matrix(model, data_loader, k_test,
                                 vit_feats, image_embeds,
                                 text_embeds, text_ids, text_atts):
    """
    Compute similarity i2t, t2i matrix for the given data loader.
    """
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset), len(data_loader.dataset)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        batch_size = 512
        for chunk_start in range(0, k_test, batch_size):
            chunk_end = chunk_start + batch_size
            chunk_indices = topk_idx[chunk_start:chunk_end]
            chunk_sim = topk_sim[chunk_start:chunk_end]

            image_inputs = vit_feats[start + i].unsqueeze(0).repeat(len(chunk_indices), 1, 1).to(model.device)

            chunk_scores = model.compute_itm(
                image_inputs=image_inputs,
                text_ids=text_ids[chunk_indices],
                text_atts=text_atts[chunk_indices],
            ).float()

            score_matrix_i2t[start + i, chunk_indices] = chunk_scores + chunk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(data_loader.dataset), len(data_loader.dataset)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)        
        batch_size = 512
        for chunk_start in range(0, k_test, batch_size):
            chunk_end = chunk_start + batch_size
            chunk_indices = topk_idx[chunk_start:chunk_end]
            chunk_sim = topk_sim[chunk_start:chunk_end]

            image_inputs = vit_feats[chunk_indices.cpu()].to(model.device)
            chunk_scores = model.compute_itm(
                image_inputs=image_inputs,
                text_ids=text_ids[start + i].repeat(len(chunk_indices), 1),
                text_atts=text_atts[start + i].repeat(len(chunk_indices), 1),
            ).float()
            score_matrix_t2i[start + i, chunk_indices] = chunk_scores + chunk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
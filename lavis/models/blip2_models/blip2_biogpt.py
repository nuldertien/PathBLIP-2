"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, BioGptForCausalLM
import transformers
import json
import os

@registry.register_model("blip2_biogpt")
class Blip2BioGPT(Blip2Base):
    """
    BLIP2 BioGPT model.
    Supported model types:
        - biogpt: BioGPT model.
        - BioGPT-Large: BioGPT-Large model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_biogpt": "configs/models/blip2/blip2_pretrain_biogpt.yaml",
        "pretrain_biogpt_large": "configs/models/blip2/blip2_pretrain_biogpt_large.yaml",
    }

    def __init__(
        self,
        vit_model='identity',
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        biogpt_model="microsoft/biogpt",
        unfreeze_last_layer=True,
        prompt="",
        max_txt_len=1024,
        apply_lemmatizer=False,
        technique=None,
        padding_per_batch=True,
        max_patches=None,
        all_info_bool=False
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.padding_per_batch = padding_per_batch
        self.all_info_bool = all_info_bool

        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 BioGPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()
        if self.padding_per_batch:
            self.padder, _ = self.init_vision_encoder(
                model_name="padding_per_batch", 
                img_size=None,
                drop_path_rate=None,
                use_grad_checkpoint=None,
                precision=None,
                technique=None,
                max_patches=max_patches
            )
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, 
            vit_precision, technique, max_patches
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.biogpt_tokenizer = AutoTokenizer.from_pretrained(
            biogpt_model, 
            use_fast=False,
        )
        self.biogpt_model = BioGptForCausalLM.from_pretrained(
            biogpt_model, 
            torch_dtype=torch.float32,
        )
        for name, param in self.biogpt_model.named_parameters():
            param.requires_grad = False

        # from peft import LoraConfig

        # lora_config = LoraConfig(
        #     r=8,                # Low-rank dimension
        #     lora_dropout=0.1,
        #     task_type="CAUSAL_LM",  # Task type (causal language modeling)
        #     target_modules=["q_proj", "v_proj"] 
        # )

        # from peft import get_peft_model

        # self.biogpt_model = get_peft_model(self.biogpt_model, lora_config)

        if unfreeze_last_layer:
            for param in self.biogpt_model.output_projection.parameters():
                param.requires_grad = True
        
        for param in self.biogpt_model.output_projection.parameters():
            print(f"The last layer of BioGPT is trainable: {param.requires_grad}")
            
        self.eos_token_id = self.biogpt_tokenizer.eos_token_id
        self.biogpt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.biogpt_model.config.hidden_size
        )
        print(self.Qformer.config.hidden_size)
        print(self.biogpt_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.biogpt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

    def forward(self, samples):
        image = samples["image"]

        with self.maybe_autocast():
            if self.padding_per_batch:
                image_embeds, image_atts = self.padder(image)
                image_embeds = self.ln_vision(self.visual_encoder(image_embeds))
            else:
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image_embeds.device
                )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Image that is encoded to input for biogpt with full attention
        inputs_biogpt = self.biogpt_proj(query_output.last_hidden_state)
        atts_biogpt = torch.ones(inputs_biogpt.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # The text contains the prompt during the forward pass, therefore 
        # seperate the prompt from the text_input for now
        prompt_length_char = len(self.prompt)
        prompts  = [t[:prompt_length_char] for t in samples['text_input']]
        texts = [t[prompt_length_char:] for t in samples['text_input']]
        texts_full_info = [t[prompt_length_char:] for t in samples['text_input_all']]

        if self.all_info_bool:
            # Calculate the max length of the text (in general: 512 - 32 - prompt length)
            maximum_length = self.max_txt_len - query_tokens.size(1) - self.prompt_length
            texts = self.custom_truncate(texts_full_info, texts, 
                                        maximum_length, self.biogpt_tokenizer)
            text = [f"{prompts[0]}{t} {self.biogpt_tokenizer.eos_token}" for t in texts]
        else:
            # The text contains the prompt during the forward pass, 
            # whereas in the generate method, it is not automatically added.
            text = [f"{t} {self.biogpt_tokenizer.eos_token}" for t in samples['text_input']]

        maximum_length = self.max_txt_len - query_tokens.size(1) # - self.prompt_length
        
        self.biogpt_tokenizer.padding_side = "right"
        biogpt_tokens = self.biogpt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=maximum_length,
            add_special_tokens=False,
        ).to(image_embeds.device)

        for i in range(len(biogpt_tokens.input_ids)):
            if biogpt_tokens.attention_mask[i].sum() == maximum_length:
                biogpt_tokens.input_ids[i][-1] = self.biogpt_tokenizer.eos_token_id
                biogpt_tokens.attention_mask[i][-1] = 1

        targets = biogpt_tokens.input_ids.masked_fill(
            biogpt_tokens.input_ids == self.biogpt_tokenizer.pad_token_id, -100
        )

        if self.prompt:
            targets[:, : self.prompt_length - 1] = -100

        sep_token = torch.tensor(
                [[self.biogpt_tokenizer.sep_token_id]] * image_embeds.size(0)
            ).to(image_embeds.device)
        embedded_sep_token = self.biogpt_model.biogpt.embed_tokens(
            sep_token
        )

        atts_sep_token = torch.tensor(
            [[1]] * image_embeds.size(0)
        ).to(image_embeds.device)
        target_sep_token = torch.tensor(
            [[-100]] * image_embeds.size(0)
        ).to(image_embeds.device)

        empty_targets = (
            torch.ones(atts_biogpt.size(), dtype=torch.long).to(image_embeds.device).fill_(-100)
        )
        inputs_embeds = self.biogpt_model.biogpt.embed_tokens(biogpt_tokens.input_ids)
        
        inputs_embeds = torch.cat([embedded_sep_token, inputs_biogpt, embedded_sep_token, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_sep_token, atts_biogpt, atts_sep_token, biogpt_tokens.attention_mask], dim=1)
        targets = torch.cat([target_sep_token, empty_targets, sep_token, targets], dim=1)

        with self.maybe_autocast():
            outputs = self.biogpt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=8,
        max_length=1024,
        min_length=50,
        top_p=1.0,
        repetition_penalty=1.2,
        length_penalty=1.1,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        with self.maybe_autocast():
            if self.padding_per_batch:
                image_embeds, image_atts = self.padder(image)
                image_embeds = self.ln_vision(self.visual_encoder(image_embeds))
            else:
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image_embeds.device
                )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_biogpt = self.biogpt_proj(query_output.last_hidden_state)
            atts_biogpt = torch.ones(inputs_biogpt.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            if prompt.strip() != "":

                prompt = [prompt] * image_embeds.size(0)
            
                biogpt_tokens = self.biogpt_tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    max_length=None,
                    add_special_tokens=False
                ).to(image_embeds.device)

                sep_token = self.biogpt_model.biogpt.embed_tokens(
                        torch.tensor(
                            [[self.biogpt_tokenizer.sep_token_id]] * image_embeds.size(0)
                        ).to(image_embeds.device)
                    )
                atts_sep_token = torch.tensor(
                    [[1]] * image_embeds.size(0)
                ).to(image_embeds.device)
                inputs_embeds = self.biogpt_model.biogpt.embed_tokens(biogpt_tokens.input_ids)
                inputs_embeds = torch.cat([sep_token, inputs_biogpt, sep_token, inputs_embeds], dim=1)

                attention_mask = torch.cat([atts_sep_token, atts_biogpt, atts_sep_token, biogpt_tokens.attention_mask], dim=1)
            else:
                sep_token = self.biogpt_model.biogpt.embed_tokens(
                        torch.tensor(
                            [[self.biogpt_tokenizer.sep_token_id]] * image_embeds.size(0)
                        ).to(image_embeds.device)
                    )
                atts_sep_token = torch.tensor(
                    [[1]] * image_embeds.size(0)
                ).to(image_embeds.device)
                inputs_embeds = torch.cat([sep_token, inputs_biogpt, sep_token], dim=1)
                attention_mask = torch.cat([atts_sep_token, atts_biogpt, atts_sep_token], dim=1)

            outputs = self.biogpt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                # temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.biogpt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            output_text = [text.strip() for text in output_text]
            return output_text
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_biogpt = self.biogpt_proj(query_output.last_hidden_state)
            atts_biogpt = torch.ones(inputs_biogpt.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.biogpt_tokenizer.padding_side = "right"
            biogpt_tokens = self.biogpt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image_embeds.device)
        
            attention_mask = torch.cat([atts_biogpt, biogpt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.biogpt_model.get_input_embeddings()(biogpt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_biogpt,inputs_embeds],dim=1)
            
            outputs = self.biogpt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.biogpt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
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

    @torch.no_grad()
    def generate_embeddings(
        self,
        samples
        ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
        """
        image = samples["image"]
        with self.maybe_autocast():
            if self.padding_per_batch:
                image_embeds, image_atts = self.padder(image)
                image_embeds = self.ln_vision(self.visual_encoder(image_embeds))
            else:
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image_embeds.device
                )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_biogpt = self.biogpt_proj(query_output.last_hidden_state)
            
        image_embedding = {}
        for image_id, i in zip(samples["image_id"], range(len(samples["image_id"]))):
            image_embedding[image_id] = inputs_biogpt[i].cpu().numpy().tolist()

        file_path = "./embedded_image_collection.txt"

        def save_embeddings_text(file_path, image_embeddings):
            with open(file_path, "a") as f:
                for image_id, embedding in image_embeddings.items():
                    record = {"image_id": image_id, "embedding": embedding}
                    f.write(json.dumps(record) + "\n")

        save_embeddings_text(file_path, image_embedding)

        # # 1. Load existing data if the file exists; otherwise, create an empty dict
        # if os.path.exists(file_path):
        #     with open(file_path, "r") as f:
        #         accumulated_embeddings = json.load(f)
        # else:
        #     accumulated_embeddings = {}

        # # 2. Create a dict for the current batch
        # current_batch_embeddings = {}
        # for image_id, i in zip(samples["image_id"], range(len(samples["image_id"]))):
        #     current_batch_embeddings[image_id] = inputs_biogpt[i].cpu().numpy().tolist()

        # # 3. Merge current batch embeddings into the accumulated file content
        # accumulated_embeddings.update(current_batch_embeddings)

        # # 4. Write the merged embeddings back to file
        # with open(file_path, "w") as f:
        #     json.dump(accumulated_embeddings, f)

        return 
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "identity")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        biogpt_model = cfg.get("biogpt_model")
        unfreeze_last_layer = cfg.get('unfreeze_last_layer')

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        # Padding per batch
        padding_per_batch = cfg.get("padding_per_batch", False)
        if padding_per_batch and vit_model != 'identity':
            raise ValueError(
                "Padding per batch is only supported with identity model."
                )
        max_patches = cfg.get("max_patches", None)

        # Adaptive pooling
        technique = cfg.get("technique", "max")  

        # Experimental settings
        all_info_bool = cfg.get("all_info_bool", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 1024)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            biogpt_model=biogpt_model,
            unfreeze_last_layer=unfreeze_last_layer,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            technique=technique,
            padding_per_batch=padding_per_batch,
            max_patches=max_patches,
            all_info_bool=all_info_bool,
        )
        model.load_checkpoint_from_config(cfg)

        return model
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import json
import os
from lavis.common.dist_utils import main_process #, get_rank
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.utils import is_convertible_to_int #, is_url, cache_url
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import is_dist_avail_and_initialized
import torch.distributed as dist
import torch
import numpy as np


@registry.register_task("pathology_captioning")
class PathologyCaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, repetition_penalty, length_penalty, top_p, temperature, evaluate, 
                 report_metric_during_training=False, report_metric=True, split=["val"]):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.top_p = top_p
        self.temperature = temperature
        self.evaluate = evaluate
        self.report_metric_during_training = report_metric_during_training
        self.report_metric = report_metric
        assert len(split) == 1, "Only support one split for evaluation."
        self.split = split[0]
        self.prompt_length = None

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 8)
        max_len = run_cfg.get("max_len", 1024)
        min_len = run_cfg.get("min_len", 50)
        repetition_penalty = run_cfg.get("repetition_penalty", 1.2)
        length_penalty = run_cfg.get("length_penalty", 1.1) 
        top_p = run_cfg.get("top_p", 1.0)
        temperature = run_cfg.get("temperature", 0.8)
        evaluate = run_cfg.evaluate
        report_metric_during_training = run_cfg.get("report_metric_during_training", False)
        report_metric = run_cfg.get("report_metric", True)
        split = run_cfg.get("valid_splits", ["val"])

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            top_p=top_p,
            temperature=temperature,
            evaluate=evaluate,
            report_metric_during_training=report_metric_during_training,
            report_metric=report_metric,
            split=split,
        )

    def train_step(self, model, samples):
        if self.prompt_length == None:
            self.prompt_length = len(model.module.prompt)
        else:
            pass

        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v

        if self.report_metric_during_training:
            generated_captions, _ = self.valid_step(model=model.module, samples=samples, split_name="train")
            # train_metrics = self._report_metrics(
            #     eval_results=caption_generation_output, split_name="train"
            # )
        else:
            generated_captions = None
            
        return output["loss"], loss_dict, generated_captions

    def valid_step(self, model, samples, split_name="val"):
        results = []

        # model.generate_embeddings(samples)
        # return [{'caption': 'bla', 'gt_caption':'bla', 'image_id':1}], {'loss': 0}

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            top_p=self.top_p,
            # temperature=self.temperature,
        )
        
        img_ids = samples["image_id"]

        if model.all_info_bool:
            gt_captions = samples['text_input_all']
        else:
            gt_captions = samples['text_input']

        for caption, gt_caption, img_id in zip(captions, gt_captions, img_ids):
            img_id = int(img_id) if is_convertible_to_int(img_id) else img_id
            results.append({"caption": caption, "gt_caption": gt_caption[self.prompt_length:], "image_id": img_id})
        
        if split_name == "val":
            with torch.no_grad():
                eval_loss = model(samples)
        else:
            eval_loss = None

        return results, eval_loss
    
    def evaluation(self, model, data_loader, split_name=None, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50

        results = []

        # Evaluation phase for the model
        model.eval()

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output, eval_loss = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)
            metric_logger.update(**eval_loss)

        metric_logger.synchronize_between_processes()
        
        if is_dist_avail_and_initialized():
            dist.barrier()

        self.results_global = results

        if self.report_metric:
            metrics = self._report_metrics(
                eval_results=results, split_name=self.split
            )
        else:
            metrics = {'agg_metrics': 0.0}
            print("NO METRICS REPORTED")

        metric_logger.update(**metrics)
        metric_logger.synchronize_between_processes()

        print("Validation results: {}".format(metric_logger.global_avg()))
        
        metrics = {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

        return metrics

    def after_evaluation(self, split_name, epoch, **kwargs):
        self.save_result(
            result=self.results_global,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="", 
        )

    @main_process
    def _report_metrics(self, eval_results, split_name):
        # Extra code to avoid unnecessary prints, do not know how to fix it otherwise
        # The only important part of the upcoming code is `eval_res = compute_scores(...)`
        # This code supresses prints that come from Java (I believe).
        ##### FROM HERE #####
        # Redirect stderr to /dev/null
        fnull = open(os.devnull, 'w')
        original_stderr = os.dup(2)  # Duplicate the original stderr (file descriptor 2)
        os.dup2(fnull.fileno(), 2)  # Redirect stderr to /dev/null

        try:
            eval_res, eval_res_total = compute_scores({i: [gt] for i, gt in enumerate([x['gt_caption'] for x in eval_results])},
                                      {i: [re] for i, re in enumerate([x['caption'] for x in eval_results])})
        finally:
            # Restore original stderr
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            fnull.close()
        ##### UNTIL HERE #####
        agg_metrics = sum(eval_res.values())
        log_stats = {split_name: {k: v for k, v in eval_res.items()}}

        if split_name == "val":
            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
            with open(
                os.path.join(registry.get_path("output_dir"), "total_evaluation_scores.txt"), "a"
            ) as f:
                eval_res_total_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in eval_res_total.items()}
                f.write(json.dumps(eval_res_total_serializable) + "\n")

        coco_res = {k: v for k, v in eval_res.items()}
        coco_res["agg_metrics"] = agg_metrics
        return coco_res
    
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        gt_and_generated_captions = []

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            ## notify model that sample is empty (error occured)
            if not isinstance(samples, dict):
                samples = {"is_empty":True}

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict, generated_captions = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if generated_captions is not None:
                gt_and_generated_captions.extend(generated_captions)

        if self.report_metric_during_training:
            training_metrics = self._report_metrics(
                eval_results=gt_and_generated_captions, split_name="train"
            )
            metric_logger.update(**training_metrics)

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    
def compute_scores(gts, res): 
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"), # sudo apt-get install default-jre <- to install java and not return error
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE") 
    ]

    eval_res = {}
    eval_res_total = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
            for sc, m in zip(scores, method):
                eval_res_total[m] = sc
        else:
            eval_res[method] = score
            eval_res_total[method] = scores


    return eval_res, eval_res_total

# The following code is taken from the pycocoevalcap library
# It is needed to put bleu_scorer.compute_score to verbose = 0

class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)
        
        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import torch
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import is_dist_avail_and_initialized
import torch.distributed as dist
import numpy as np
import os 

@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        model.eval()
        with torch.no_grad():
            output, image, text = model.evaluation_forward(samples)

        loss_dict = {}
        for k, v in output.items():
            if "loss" in k:
                loss_dict["eval_" + str(k)] = v

        return loss_dict, image, text
    
    def evaluation(self, model, data_loader, split_name=None, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50

        text_embeds = []
        text_ids = []
        text_atts = []
        text_embeds_other = []
        text_ids_other = []
        text_atts_other = []

        vit_feats = []
        image_embeds = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            loss_dict, image_features, text_features = self.valid_step(
                model=model, samples=samples
            )

            if split_name == 'test' or split_name == 'val':
                text_embeds.append(text_features[0])
                text_ids.append(text_features[1])
                text_atts.append(text_features[2])
                text_embeds_other.append(text_features[3])
                text_ids_other.append(text_features[4])
                text_atts_other.append(text_features[5])

                vit_feats.append(image_features[0])
                image_embeds.append(image_features[1])

            metric_logger.update(**loss_dict)

        metric_logger.synchronize_between_processes()

        if is_dist_avail_and_initialized():
            dist.barrier()
        
        results = {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

        # In this case, the best model is the one with the lowest 'eval_loss'
        # We transform 'eval_loss' into a negative number since the best model is
        # chosen based on the highest value of 'agg_metrics'
        agg_metrics = {'agg_metrics':-float(results['eval_loss'])}
        # Remove 'eval_loss' from results
        results.pop('eval_loss')
        # Merge the two dictionaries, in this way 'agg_metrics' will be the first key
        results = {**agg_metrics, **results}
        # results is constructed as:
        # total_loss, loss_itc, loss_itm, loss_lm
        if split_name == 'test' or split_name == 'val':
            import sys
            sys.exit()
            
            text_embeds = torch.cat(text_embeds, dim=0)
            text_ids = torch.cat(text_ids, dim=0)
            text_atts = torch.cat(text_atts, dim=0)
            
            vit_feats = torch.cat(vit_feats, dim=0)
            image_embeds = torch.cat(image_embeds, dim=0)

            score_matrix_i2t, score_matrix_t2i = model.compute_sim_matrix(
                                    data_loader, 0, 
                                    vit_feats, image_embeds,
                                    text_embeds, text_ids, text_atts)
            
            recall = self._report_metrics(score_matrix_i2t, 
                                          score_matrix_t2i,
                                          other=False)
            
            text_embeds_other = torch.cat(text_embeds_other, dim=0)
            text_ids_other = torch.cat(text_ids_other, dim=0)
            text_atts_other = torch.cat(text_atts_other, dim=0)

            score_matrix_i2t_other, score_matrix_t2i_other = model.compute_sim_matrix(
                                    data_loader, 0, 
                                    vit_feats, image_embeds,
                                    text_embeds_other, text_ids_other, 
                                    text_atts_other)
            
            recall_other = self._report_metrics(score_matrix_i2t_other, 
                                                score_matrix_t2i_other,
                                                other=True)
            
            # Recall all keys with adding "_other" at the end
            recall_other = {k + "_other": v for k, v in recall_other.items()}

            # Merge R@K results into final results dictionary
            results = {**results, **recall, **recall_other}

            # put recall in metric_logger
            metric_logger.update(**recall)
            metric_logger.update(**recall_other)


        print("Validation results: {}".format(metric_logger.global_avg()))

        return results

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, other=False):
        """
        Computes retrieval metrics R@1,5,10,25,100, mean & median for 
        image-to-text and text-to-image retrieval.
        Args:
            scores_i2t: Similarity scores for image-to-text retrieval.
            scores_t2i: Similarity scores for text-to-image retrieval.
        Returns:
            eval_result: A dictionary containing the retrieval metrics.
        """

        logits = {
            "image_to_text": torch.from_numpy(scores_i2t),  # shape: (N, N) => images x texts
            "text_to_image": torch.from_numpy(scores_t2i)    # shape: (N, N) => texts x images
        }

        torch.save(logits, os.path.join(registry.get_path("output_dir"), f"./logits_{other}.pt"))
        
        # Generate ground-truth indices
        txt2img = np.arange(scores_t2i.shape[0])  # Assuming the i-th text corresponds to the i-th image
        img2txt = txt2img  # Assuming the i-th image corresponds to the i-th text

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]  # Sort in descending order to get ranks
            rank = np.where(inds == img2txt[index])[0][0]  # Find the ground truth index
            ranks[index] = rank

        print(ranks)
        # Compute metrics for image-to-text retrieval
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        tr25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
        tr100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

        mean_rank_i2t = np.mean(ranks) + 1
        median_rank_i2t = np.median(ranks) + 1

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])
        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]  # Sort in descending order to get ranks
            ranks[index] = np.where(inds == txt2img[index])[0][0]  # Find the ground truth index

        # Compute metrics for text-to-image retrieval
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        ir25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
        ir100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

        mean_rank_t2i = np.mean(ranks) + 1
        median_rank_t2i = np.median(ranks) + 1

        eval_result = {
            "txt_r1": tr1, 
            "txt_r5": tr5, 
            "txt_r10": tr10, 
            "txt_r25": tr25, 
            "txt_r100": tr100,
            "txt_mean_rank": mean_rank_i2t, 
            "txt_median_rank": median_rank_i2t,  
            "img_r1": ir1, 
            "img_r5": ir5, 
            "img_r10": ir10, 
            "img_r25": ir25, 
            "img_r100": ir100,
            "img_mean_rank": mean_rank_t2i, 
            "img_median_rank": median_rank_t2i  
        }

        return eval_result
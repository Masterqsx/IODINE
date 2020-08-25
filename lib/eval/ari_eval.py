from .base import Evaluator
import torch
from lib.utils.ari import compute_mask_ari
import numpy as np
from skimage import io

def to_one_hot(tensor, dim):
    one_hot = torch.zeros(tensor.shape).to(tensor.device)
    index = torch.argmax(tensor, dim=dim, keepdim=True)
    one_hot = one_hot.scatter(dim=dim, index=index, value=1)
    return one_hot
class ARIEvaluator(Evaluator):
    def __init__(self):
        Evaluator.__init__(self)
        
        self.aris = []

    def adjusted_rand_index_pytorch(self, true_mask, pred_mask):
        """
        borrowed from https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py
        true_mask: pytorch Tensor of shape [batch_size, n_points, n_true_groups]
            The true cluster assignment encoded as one-hot.
        pred_mask: pytorch Tensor of shape [batch_size, n_points, n_pred_groups]
            The predicted cluster assignment encoded as categorical probabilities.
        """
        _, n_points, n_true_groups = true_mask.shape
        n_pred_groups = pred_mask.shape[-1]

        if n_true_groups == n_pred_groups and n_true_groups == 1:
            return 1.

        if n_points <= n_true_groups and n_points <= n_pred_groups:
            raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint.")

        true_mask_oh = torch.as_tensor(true_mask, dtype=torch.float32)
        pred_mask_oh = torch.as_tensor(to_one_hot(pred_mask, 2), dtype=torch.float32)

        n_points = torch.as_tensor(true_mask_oh.sum(dim=[1, 2]), dtype=torch.float32)
        nij = torch.bmm(true_mask_oh.permute(0, 2, 1), pred_mask_oh)
        a = nij.sum(dim=1)
        b = nij.sum(dim=2)

        # !!! Unknown why nij * (nij - 1.) has difference with tensorflow implementation
        # While nij is the same as tensorflow implementation
        # Always have 0.02 difference in ARI
        rindex = torch.sum(nij * (nij - 1.), dim=(1, 2))
        aindex = torch.sum(a * (a - 1.), dim=1)
        bindex = torch.sum(b * (b - 1.), dim=1)
        expected_rindex = aindex * bindex / (n_points * (n_points - 1.))
        max_rindex = (aindex + bindex) / 2.
        ari = ((rindex - expected_rindex) / (
                    max_rindex - expected_rindex + torch.finfo(torch.float32).eps)).mean().item()

        # temp = nij.numpy()
        return ari
        
    def evaluate(self, model, data):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            mask: list, each is (N, H, W)
        :return: average ari
        """
        from torch import arange as ar
        image, mask = data
        pred, pred_mask, mean = model.reconstruct(image)
        # (B, K, 1, H, W)
        # (B, K, H, W)
        pred_mask = pred_mask[:, :, 0]
        
        B, K, H, W = pred_mask.size()

        # reduced to (B, K, H, W), with 1-0 values

        # max_index (B, H, W)
        # max_index = torch.argmax(pred_mask, dim=1)
        # get binarized masks (B, K, H, W)
        # pred_mask = torch.zeros_like(pred_mask)
        # pred_mask[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0

        # for b in range(B):
        this_ari = self.adjusted_rand_index_pytorch(torch.stack(mask).cuda()[:, 1:, :, :].flatten(start_dim=2).permute(0, 2, 1), pred_mask.detach().flatten(start_dim=2).permute(0, 2, 1))
        # for k in range(K):
        #     io.imsave(f"m_{k}.jpg", pred_mask[0, k].detach().cpu().numpy())
        # io.imsave(f"im.jpg", image[0].cpu().permute(1, 2, 0).numpy())
        self.aris.append(this_ari)


    def reset(self):
        self.aris = []
    
    def get_results(self):
        return 'Ari: {}'.format(np.mean(self.aris) if self.aris else 0)

    def get_result_dict(self):
        return dict(ari=np.mean(self.aris) if self.aris else 0)

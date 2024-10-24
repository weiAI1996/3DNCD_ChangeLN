# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.utils import resize
from opencd.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from torch import Tensor
from typing import List, Optional
@MODELS.register_module()
class APDEncoderDecoder(BaseSegmentor):
    """APD Encoder Decoder segmentors.
    APDEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor: OptConfigType = None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        backbone_inchannels = 3
        img1, img2 = torch.split(img, backbone_inchannels, dim=1)
        x = self.backbone(img1, img2)

        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    # def encode_decode(self, inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     x = self.extract_feat(inputs)
    #     seg_logits = self.decode_head.predict(x, batch_img_metas,
    #                                           self.test_cfg)

    #     return seg_logits
    # def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for decode head in
    #     training."""
    #     losses = dict()
    #     loss_decode = self.decode_head.forward_train(x, img_metas,
    #                                                  gt_semantic_seg,
    #                                                  self.train_cfg)

    #     losses.update(add_prefix(loss_decode, 'decode'))
    #     return losses
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)
    

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    # def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     if isinstance(self.auxiliary_head, nn.ModuleList):
    #         for idx, aux_head in enumerate(self.auxiliary_head):
    #             loss_aux = aux_head.forward_train(x, img_metas,
    #                                               gt_semantic_seg,
    #                                               self.train_cfg)
    #             losses.update(add_prefix(loss_aux, f'aux_{idx}'))
    #     else:
    #         loss_aux = self.auxiliary_head.forward_train(
    #             x, img_metas, gt_semantic_seg, self.train_cfg)
    #         losses.update(add_prefix(loss_aux, 'aux'))

    #     return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
    
    # def forward_train(self, img, img_metas, gt_semantic_seg):
    #     """Forward function for training.
    #     Args:
    #         img (Tensor): Input images.
    #         img_metas (list[dict]): List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:Collect`.
    #         gt_semantic_seg (Tensor): Semantic segmentation masks
    #             used if the architecture supports semantic segmentation task.
    #     Returns:
    #         dict[str, Tensor]: a dictionary of loss components
    #     """

    #     x = self.extract_feat(img)

    #     losses = dict()

    #     loss_decode = self._decode_head_forward_train(x, img_metas,
    #                                                   gt_semantic_seg)
    #     losses.update(loss_decode)

    #     if self.with_auxiliary_head:
    #         loss_aux = self._auxiliary_head_forward_train(
    #             x, img_metas, gt_semantic_seg)
    #         losses.update(loss_aux)

    #     return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    # def whole_inference(self, inputs: Tensor,
    #                     batch_img_metas: List[dict]) -> Tensor:
    #     """Inference with full image.

    #     Args:
    #         inputs (Tensor): The tensor should have a shape NxCxHxW, which
    #             contains all images in the batch.
    #         batch_img_metas (List[dict]): List of image metainfo where each may
    #             also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
    #             'ori_shape', and 'pad_shape'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

    #     Returns:
    #         Tensor: The segmentation results, seg_logits from model of each
    #             input image.
    #     """

    #     seg_logits = self.encode_decode(inputs, batch_img_metas)

    #     return seg_logits

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    # TODO refactor
    # def slide_inference(self, img, img_meta, rescale):
    #     """Inference by sliding-window with overlap.
    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.
    #     """

    #     h_stride, w_stride = self.test_cfg.stride
    #     h_crop, w_crop = self.test_cfg.crop_size
    #     batch_size, _, h_img, w_img = img.size()
    #     num_classes = self.num_classes
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]
    #             crop_seg_logit = self.encode_decode(crop_img, img_meta)
    #             preds += F.pad(crop_seg_logit,
    #                            (int(x1), int(preds.shape[3] - x2), int(y1),
    #                             int(preds.shape[2] - y2)))

    #             count_mat[:, :, y1:y2, x1:x2] += 1
    #     assert (count_mat == 0).sum() == 0
    #     if torch.onnx.is_in_onnx_export():
    #         # cast count_mat to constant while exporting to ONNX
    #         count_mat = torch.from_numpy(
    #             count_mat.cpu().detach().numpy()).to(device=img.device)
    #     preds = preds / count_mat
    #     if rescale:
    #         preds = resize(
    #             preds,
    #             size=img_meta[0]['ori_shape'][:2],
    #             mode='bilinear',
    #             align_corners=self.align_corners,
    #             warning=False)
    #     return preds

    # def whole_inference(self, img, img_meta, rescale):
    #     """Inference with full image."""

    #     seg_logit = self.encode_decode(img, img_meta)
    #     if rescale:
    #         # support dynamic shape for onnx
    #         if torch.onnx.is_in_onnx_export():
    #             size = img.shape[2:]
    #         else:
    #             size = img_meta[0]['ori_shape'][:2]
    #         seg_logit = resize(
    #             seg_logit,
    #             size=size,
    #             mode='bilinear',
    #             align_corners=self.align_corners,
    #             warning=False)

    #     return seg_logit

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    # def inference(self, img, img_meta, rescale):
    #     """Inference with slide/whole style.
    #     Args:
    #         img (Tensor): The input image of shape (N, 3, H, W).
    #         img_meta (dict): Image info dict where each dict has: 'img_shape',
    #             'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             `mmseg/datasets/pipelines/formatting.py:Collect`.
    #         rescale (bool): Whether rescale back to original shape.
    #     Returns:
    #         Tensor: The output segmentation map.
    #     """

    #     assert self.test_cfg.mode in ['slide', 'whole']
    #     ori_shape = img_meta[0]['ori_shape']
    #     assert all(_['ori_shape'] == ori_shape for _ in img_meta)
    #     if self.test_cfg.mode == 'slide':
    #         seg_logit = self.slide_inference(img, img_meta, rescale)
    #     else:
    #         seg_logit = self.whole_inference(img, img_meta, rescale)
    #     output = F.softmax(seg_logit, dim=1)
    #     flip = img_meta[0]['flip']
    #     if flip:
    #         flip_direction = img_meta[0]['flip_direction']
    #         assert flip_direction in ['horizontal', 'vertical']
    #         if flip_direction == 'horizontal':
    #             output = output.flip(dims=(3, ))
    #         elif flip_direction == 'vertical':
    #             output = output.flip(dims=(2, ))

    #     return output

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
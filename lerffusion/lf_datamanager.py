
"""
Lerffusion Datamanager.
TODO: 
- ray sampling @david 
- crop imgs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

import numpy as np
# Our own imports
import torch
#import PIL
from transformers import YolosFeatureExtractor, YolosForObjectDetection
#import torch
import yolov5
import PIL

CONSOLE = Console(width=120)

@dataclass
class LerffusionDataManagerConfig(VanillaDataManagerConfig):

    _target: Type = field(default_factory=lambda: LerffusionDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class LerffusionDataManager(VanillaDataManager):

    config: LerffusionDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )

        #TODO: only sample rays that pass through the lerf mask
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch['image'] = self.image_batch['image'].clone()
        self.original_image_batch['image_idx'] = self.image_batch['image_idx'].clone()

        self.yolo = self.load_yolo5_model()
        #self.yolo_feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
        #self.yolo_obj = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

        #print(type(self.original_image_batch['image'])) 
        #print(self.original_image_batch['image'].shape)
        


        #self.mask_images = [self.get_mask(img) for img in self.original_image_batch['image']]
        #self.mask_images = self.get_mask(self.original_image_batch['image'].permute(0,3,1,2))
        self.mask_images = self.get_mask([img.numpy() for img in self.original_image_batch['image']])


    def load_yolo5_model(self):
        """
        Returns:
            Yolo5 model. see https://github.com/fcakyon/yolov5-pip for documentation
        """
        model = yolov5.load('yolov5s.pt')
        model.conf = 0.25  # NMS confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        model.max_det = 20  # maximum number of detections per image
        return model

    def get_mask(self, imgs):

        #inputs = self.yolo_feature_extractor(images=img, return_tensors="pt")
        #outputs = self.yolo_obj(**inputs)
        test = imgs[0] * 255
        im = PIL.Image.fromarray(test.astype(np.uint8))
        im.save("/n/fs/nlp-jiatongy/lerffusion/debug.png")
        # for img in imgs: 
        #     print(img.shape)
        pass_imgs = [img*255 for img in imgs]
        preds = self.yolo(pass_imgs).xyxy #numpy form

        boxes_list = [pred[:, :4] for pred in preds]
        categories_list = [pred[:, 5] for pred in preds] 
        # print(categories_list)       
        # for idx, ctr in enumerate(categories_list): 
        #     print((ctr==47).nonzero(as_tuple=True))
        #     if len((ctr==47).nonzero(as_tuple=True)[0]) == 0: 
        #         print("empty")
        #         im = PIL.Image.fromarray((imgs[idx]*255).astype(np.uint8))
        #         im.save("/n/fs/nlp-jiatongy/lerffusion/n_debug.png")
        mask_imgs = []
        for idx, ctr in enumerate(categories_list): 
            # empty 
            img = imgs[idx]
            if len((ctr==47).nonzero(as_tuple=True)[0]) == 0: 
                mask_img = torch.unsqueeze(torch.zeros(img[:,:,0].shape),dim=-1)
            else: 
                bbox = boxes_list[idx][int((ctr==47).nonzero(as_tuple=True)[0][0])].int()
                mask_img = torch.unsqueeze(torch.zeros(img[:,:,0].shape),dim=-1)
                mask_img[bbox[1]-50:bbox[3]+50, bbox[0]-50:bbox[2]+50] = 255
            mask_imgs.append(mask_img)

        mask_imgs = torch.stack(mask_imgs, dim=0).permute(0,-1,1,2)

        return mask_imgs

        # predictions = self.yolo(img).pred[0]

        # box = predictions[int((predictions[:,5] == 47).nonzero(as_tuple=True)[0])][:4] #x1,y1,x2,y2
        # mask = np.zeros(img.shape)
        # #TODO @jiatong do we really need to be using cv2 here?
        # mask = cv2.rectangle(mask,pt1=(int(box[0]),int(box[1])),pt2=(int(box[2]),int(box[3])),color=(0,)*3,thickness=-1)
    
    def get_batch_masks(self, image_batch):

        images = image_batch['image'].clone()
        if isinstance(images, list):
            masks = torch.cat([self.get_mask(img) for img in images], dim=0)
            return masks
        else:
            return self.get_mask(images)[None, :, :, :] #TODO: does this need to be a list?

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None


        #mask_images = self.get_batch_masks(self.image_batch)
        #self.image_batch["mask"] = mask_images

        batch = self.train_pixel_sampler.sample(self.image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        return ray_bundle, batch

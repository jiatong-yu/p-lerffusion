"""Lerffusion Pipeline and trainer"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torchvision import transforms
from typing_extensions import Literal
from diffusers import StableDiffusionInpaintPipeline
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from lerffusion.lf_datamanager import (
    LerffusionDataManagerConfig,
)
from in2n.ip2p import InstructPix2Pix


@dataclass
class LerffusionPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LerffusionPipeline)
    """target class to instantiate"""
    datamanager: LerffusionDataManagerConfig = LerffusionDataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    edit_rate: int = 10
    """how many NeRF steps before image edit"""
    edit_count: int = 1
    """how many images to edit per NeRF step"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.02
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""

    #TODO: change to support lerf 
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = True
    """Whether to use full precision for InstructPix2Pix"""

class LerffusionPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: LerffusionPipelineConfig

    def __init__(
        self,
        config: LerffusionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # select device for InstructPix2Pix

        # =============
        """
        TODO: change these to support lerf / other masking methods
        """
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )


        #self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        pipe = pipe.to(self.device)
        self.pipe = pipe

        #self.tran = transforms.Normalize([0.5],[0.5])


        # load base text embedding using classifier free guidance
        # self.text_embedding = self.pipe._encode_prompt(
        #     self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        # )
        # =============

        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

        # viewer elements
        self.prompt_box = ViewerText(name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback)
        self.guidance_scale_box = ViewerNumber(name="Text Guidance Scale", default_value=self.config.guidance_scale, cb_hook=self.guidance_scale_callback)
        self.image_guidance_scale_box = ViewerNumber(name="Image Guidance Scale", default_value=self.config.image_guidance_scale, cb_hook=self.image_guidance_scale_callback)


    def guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for guidance scale slider"""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for text guidance scale slider"""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Callback for prompt box, change prompt in config and update text embedding"""
        self.config.prompt = handle.value
        
        # self.text_embedding = self.ip2p.pipe._encode_prompt(
        #     self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        # )

    def edit_img(self, input_pic, input_mask):
        # input_pic = transforms.functional.to_pil_image(tensor_img.permute(-1,0,1),"RGB")
        # input_mask = transforms.functional.to_pil_image(mask_img.permute(-1,0,1),"L")

        image = self.pipe(prompt="a glass of red wine on the table", 
             image=input_pic.resize([512,512]), 
             mask_image=input_mask.resize([512,512]),
             guidance_scale = 50,
            ).images[0]
        
        return image
    
    # def norm_img(self, img):
    #     n_img = self.tran(img/255.0)
    #     return n_img

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        #TODO: this ray_bundle should only sample rays that pass through the mask 
        ray_bundle, batch = self.datamanager.next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        # edit an image every ``edit_rate`` steps
        if (step % self.config.edit_rate == 0):

            # edit ``edit_count`` images in a row
            for i in range(self.config.edit_count):

                # iterate through "spot in dataset"
                current_spot = next(self.train_indices_order)
                
                # get original image from dataset
                original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
                # generate current index in datamanger
                current_index = self.datamanager.image_batch["image_idx"][current_spot]

                #mask_image = self.datamanager.mask_images[current_spot]
                mask_image = self.datamanager.mask_images[current_spot,:,:,:]

                # get current camera, include camera transforms from original optimizer
                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

                # get current render of nerf
                original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
                # print(f"rendered image shape: {rendered_image.shape}")
                # temp_pic = transforms.functional.to_pil_image(torch.squeeze(rendered_image,dim=0), "RGB")
                # temp_pic.save("rendered_img.png")
                #TODO: do we need this??
                #mask_image = mask_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

                # delete to free up memory
                del camera_outputs
                del current_camera
                del current_ray_bundle
                del camera_transforms
                torch.cuda.empty_cache()

                #input_pic = transforms.functional.to_pil_image(tensor_img.permute(-1,0,1),"RGB")
                # HEIGHT_MID = rendered_image.shape[2] // 2
                # WIDTH_MID = rendered_image.shape[3] // 2
                # CROP_MID = 350
                # h_lo =  HEIGHT_MID - CROP_MID
                # h_hi = HEIGHT_MID + CROP_MID
                # w_lo = WIDTH_MID - CROP_MID
                # w_hi = WIDTH_MID + CROP_MID
                # print(f"{h_lo} {h_hi} {w_lo} {w_hi}")
                # input_pic = rendered_image.clone()[:,:,h_lo:h_hi,w_lo:w_hi]
                # print("input pic shape after crop: ", str(input_pic.shape))

                input_pic = transforms.functional.to_pil_image(torch.squeeze(rendered_image,dim=0), "RGB")
                input_mask = transforms.functional.to_pil_image(torch.squeeze(mask_image,dim=0),"L")
                # input_pic.save("input_pic.png")
                # input_mask.save("input_mask.png")
                 
                edited_image = self.edit_img(input_pic, input_mask)
                # edited_image = rendered_image.clone()
                #edited_image[:,:,h_lo:h_hi,w_lo:w_hi] = edited_image_cropped
                # edited_image.resize(input_pic.size).save("edited_image.png")
                edited_image = transforms.ToTensor()(edited_image).unsqueeze_(0)
                
                '''
                edited_image = self.ip2p.edit_image(
                            self.text_embedding.to(self.ip2p_device),
                            rendered_image.to(self.ip2p_device),
                            mask_image.to(self.ip2p_device),
                            original_image.to(self.ip2p_device),
                            guidance_scale=self.config.guidance_scale,
                            image_guidance_scale=self.config.image_guidance_scale,
                            diffusion_steps=self.config.diffusion_steps,
                            lower_bound=self.config.lower_bound,
                            upper_bound=self.config.upper_bound,
                        )
                '''

                # resize to original image size (often not necessary)
                if (edited_image.size() != rendered_image.size()):
                    edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

                # write edited image to dataloader
                self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

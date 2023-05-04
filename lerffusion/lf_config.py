"""
Lerffusion configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from lerffusion.lf_datamanager import LerffusionDataManagerConfig
from lerffusion.lf import LerffusionModelConfig
from lerffusion.lf_pipeline import LerffusionPipelineConfig
from lerffusion.lf_trainer import LerffusionTrainerConfig

lerffusion_method = MethodSpecification(
    config=LerffusionTrainerConfig(
        method_name="lerffusion",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline= LerffusionPipelineConfig(
            datamanager=LerffusionDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LerffusionModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            ip2p_use_full_precision=True
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Lerffusion primary method: uses LPIPS, IP2P at full precision",
)

lerffusion_method_small = MethodSpecification(
    config=LerffusionTrainerConfig(
        method_name="lerffusion-small",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=LerffusionPipelineConfig(
            datamanager=LerffusionDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LerffusionModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            #TODO: change this 
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Lerffusion small method, uses LPIPs, IP2P at half precision",
)

# in2n_method_tiny = MethodSpecification(
#     config=InstructNeRF2NeRFTrainerConfig(
#         method_name="in2n-tiny",
#         steps_per_eval_batch=1000,
#         steps_per_eval_image=100,
#         steps_per_save=250,
#         max_num_iterations=30000,
#         save_only_latest_checkpoint=True,
#         mixed_precision=True,
#         pipeline=InstructNeRF2NeRFPipelineConfig(
#             datamanager=InstructNeRF2NeRFDataManagerConfig(
#                 dataparser=NerfstudioDataParserConfig(),
#                 train_num_rays_per_batch=4096,
#                 eval_num_rays_per_batch=4096,
#                 patch_size=1,
#                 camera_optimizer=CameraOptimizerConfig(
#                     mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
#                 ),
#             ),
#             model=InstructNeRF2NeRFModelConfig(
#                 eval_num_rays_per_chunk=1 << 15,
#                 use_lpips=False,
#             ),
#             ip2p_use_full_precision=False,
#         ),
#         optimizers={
#             "proposal_networks": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#             "fields": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
#                 "scheduler": None,
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Instruct-NeRF2NeRF tiny method, does not use LPIPs, IP2P at half precision",
# )
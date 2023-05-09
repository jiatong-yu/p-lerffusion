# LerfFusion 
In this project, we propose the task of natural-language-guided object editing in NeRF-based novel view synthesis. We uses [Nerfacto](https://docs.nerf.studio/en/latest/nerfology/methods/nerfacto.html) as the base model to update, and uses [stable difussion inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) to perform dataset-NeRF dual updating.

https://user-images.githubusercontent.com/71976949/235790951-d6a14b22-e785-43c9-bfae-c711b7554ed8.mp4

### Getting Started  
We use NerfStudio for integration pipelining. Refer to [this documentation](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for more details. Follow [this link](https://docs.nerf.studio/en/latest/quickstart/installation.html) to install environment and dependencies.  
1. After installing Nerfstudio, run `ns-train nerfacto` to train a vanilla NeRF model as the base model for our method.  
2. Run the following commands  
```bash
git clone https://github.com/jiatong-yu/p-lerffusion
cd lerffusion
pip install --upgrade pip setuptools
pip install -e .
```
You should be able to see `lerffusion` and `lerffusion-tiny` modes available when running `ns-train --help` after this step.  

### Training Lerffusion
Run the following commands based on the number of GPUs available. 
```bash
PROCESSED_DATA_DIR="(your dataset path)" 
LOAD_DIR="/(your trained nerfacto path)/nerfstudio_models"  
ns-train lerffusion \
          --data ${PROCESSED_DATA_DIR} \
          --load-dir ${LOAD_DIR} \
          --vis vandb \
          --pipeline.guidance-scale 50 \
          --machine.num-gpus 4 \
          --optimizers.fields.optimizer.lr 20e-4 \
```
### Limitations 
Since Stable Diffusion pipeline is wrapped as `nn.Module` in the current version of Nerfstudio, we are not able to run ns-viewer command. We will fix this bug by re-wrapping the diffuser pipeline.


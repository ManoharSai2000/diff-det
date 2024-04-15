import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from aesthetic_scorer import AestheticScorerDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from torchvision import models
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
import datetime
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags

from torchvision.transforms import v2


from dataset import ImageNette
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/align_prop.py", "Training configuration.")
from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)



def evaluate(classifier,dataloader,accelerate):
    eval_loss, eval_acc = [], 0
    num_samples = 0
    print("------Evaluating---------")
    
    for images,labels in tqdm(dataloader):
        num_samples += labels.shape[0]
        images = images.to(accelerate.device)
        labels = labels.to(accelerate.device)
        
        logits = classifier(images)
        loss = torch.nn.functional.cross_entropy(logits,labels)
        eval_loss.append(loss.cpu().numpy())
        eval_acc += (torch.argmax(logits,dim=1) == labels).sum().float().cpu().numpy()
    
    return np.mean(eval_loss),eval_acc/num_samples

    
    

def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    
    if accelerator.is_main_process:
        wandb_args = {}
        if config.debug:
            wandb_args = {'mode':"disabled"}        
        accelerator.init_trackers(
            project_name="align-prop", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, wandb.run.name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    
    # disable safety checker
    pipeline.safety_checker = None    
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.    
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16    

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device,dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device,dtype=inference_dtype)

    pipeline.unet.to(accelerator.device,dtype=inference_dtype)

    unet = pipeline.unet
    #********** classifier **********
    classifier = models.resnet18(pretrained=True)
    classifier.requires_grad_(True)

    classifier = classifier.to(accelerator.device,) #dtype=inference_dtype)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    # ******** classifier ***********
    cls_optimizer = optimizer_cls(
        classifier.parameters(),
        lr=config.train.cls_learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_fn = getattr(prompts_file, config.prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext
    
    #*********** classifier ************
    classifier, cls_optimizer = accelerator.prepare(classifier, cls_optimizer)
    transforms = torchvision.transforms.Compose([
            v2.Resize((224,224)),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    


    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    timesteps = pipeline.scheduler.timesteps
    
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0 
       
    global_step = 0

    eval_dataset = ImageNette()
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,batch_size=config.train.batch_size_per_gpu)

    if config.only_eval:
        #################### EVALUATION ONLY ####################                
        with torch.no_grad():
            eval_loss,eval_acc = evaluate(classifier,eval_dataloader,accelerator)
        print("Evaluation results", eval_acc)
        if accelerator.is_main_process:
            if config.run_name != "":
                name_val = config.run_name
            else:
                name_val = wandb.run.name            
            log_dir = f"logs/{name_val}/eval_vis"
            os.makedirs(log_dir, exist_ok=True)                    
            accelerator.log({"eval_loss": eval_loss,"eval_acc":eval_acc},step=global_step)        
    else:

        #################### TRAINING ####################        
        for epoch in list(range(first_epoch, config.num_epochs)):

            info = defaultdict(list)
            info_vis = defaultdict(list)
            image_vis_list = []
            
            for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_local_main_process):
                classifier.train()
                latent = torch.randn((config.train.batch_size_per_gpu_available, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)    

                if accelerator.is_main_process:

                    logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")

                
                prompts, labels = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )
                labels = torch.tensor(labels)
                labels = labels.to(accelerator.device, dtype=torch.long)

                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)   

                pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]   

                with torch.no_grad():
                    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                        t = torch.tensor([t],
                                            dtype=inference_dtype,
                                            device=latent.device)
                        t = t.repeat(config.train.batch_size_per_gpu_available)
                        
                        if config.grad_checkpoint:
                            noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                            noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                        else:
                            noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                            noise_pred_cond = unet(latent, t, prompt_embeds).sample
                                                        
                        if config.truncated_backprop:
                            if config.truncated_backprop_rand:
                                timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                if i < timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()
                            else:
                                if i < config.trunc_backprop_timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()

                        grad = (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad                
                        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                            
                    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                            
                with accelerator.accumulate(classifier):
                    with autocast():
                        with torch.enable_grad(): # important b/c don't have on by default in module                        

                            viz_ims = ims.clone().detach()

                            ims = (ims/ 2 + 0.5).clamp(0, 1)
                            ims = transforms(ims)
                            logits = classifier(ims)

                            loss = loss_fn(logits,labels)
                            
                            loss =  loss.sum()
                            loss = loss/config.train.batch_size_per_gpu_available
                            loss = loss * config.train.loss_coeff

                            accuracy = (torch.argmax(logits,dim=1) == labels).float()
                            
                            if len(info_vis["image"]) < config.max_vis_images:
                                info_vis["image"].append(viz_ims)
                                info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                                info_vis["acc"].append(accuracy.clone().detach())
                            
                            info["loss"].append(loss)
                            info["accuracy"].append(accuracy.mean())
                            
                            # backward pass
                            accelerator.backward(loss)

                            #*******classifier**********
                            cls_optimizer.step()
                            cls_optimizer.zero_grad()
                     

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0
                    # log training and evaluation 
                    if config.visualize_eval and (global_step % config.vis_freq ==0):

                        with torch.no_grad():
                            classifier.eval()
                            eval_loss,eval_acc = evaluate(classifier,eval_dataloader,accelerator)
                            # info["eval_loss"].append(eval_loss)
                            # info["eval_accuracy"].append(eval_acc)
                            

                            if accelerator.is_main_process:
                                if config.run_name != "":
                                    name_val = config.run_name
                                else:
                                    name_val = wandb.run.name            
                                log_dir = f"logs/{name_val}/eval_vis"
                                os.makedirs(log_dir, exist_ok=True)                    
                                accelerator.log({"eval_loss": eval_loss,"eval_acc":eval_acc},step=global_step)
                    
                    logger.info("Logging")
                    
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, accuracy: {info['accuracy']}")
                    #logger.info(f"eval_loss: {info['eval_loss']}, eval_acc: {info['eval_accuracy']}")


                    info.update({"epoch": epoch, "inner_epoch": inner_iters,"eval_accuracy":eval_acc,"eval_loss":eval_loss})
                    accelerator.log(info, step=global_step)

                    if config.visualize_train:
                        ims = torch.cat(info_vis["image"])
                        acc = torch.cat(info_vis["acc"])
                        prompts = info_vis["prompts"]
                        images  = []
                        for i, image in enumerate(ims):
                            image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            pil = pil.resize((256, 256))
                            prompt = prompts[i]
                            ac = acc[i]
                            images.append(wandb.Image(pil, caption=f"{prompt:.25} | {ac:.2f}"))
                        
                        accelerator.log(
                            {"images": images},
                            step=global_step,
                        )

                    global_step += 1
                    info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients
            
            # if epoch % config.save_freq == 0 and accelerator.is_main_process:
            #     # accelerator.save_state()
            #     output_splits = output_dir.split("/")
            #     output_splits[1] = wandb.run.name
            #     output_dir = "/".join(output_splits)
            #     # accelerator.save_model(classifier, output_dir)


if __name__ == "__main__":
    app.run(main)

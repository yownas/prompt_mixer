# Prompt mixer script for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/yownas/prompt_mixer
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.

import json
import math
import os
import sys
import re
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
import cv2
from skimage import exposure
import modules.scripts as scripts
import gradio as gr
import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.face_restoration
import modules.images as images
import modules.styles
import logging
from modules.processing import Processed, process_images, fix_seed, StableDiffusionProcessing, get_fixed_seed, create_infotext
from modules.shared import opts, cmd_opts, state
from modules.prompt_parser import ScheduledPromptConditioning, MulticondLearnedConditioning, ComposableScheduledPromptConditioning


class Script(scripts.Script):
    def title(self):
        return "Prompt mixer"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        prompt2 = gr.Textbox(label='Second prompt', value='')
        weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Interpolation Amount (First prompt <-> Second prompt)', value=0.5)

        return [prompt2, weight]

    ##############

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def run(self, p, prompt2, weight):

        def prompt_mix_process_images(p: StableDiffusionProcessing, latent) -> Processed:
            """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
            # FIXME remove? 
            if type(p.prompt) == list:
                assert(len(p.prompt) > 0)
            else:
                assert p.prompt is not None
    
            devices.torch_gc()
    
            seed = get_fixed_seed(p.seed)
            subseed = get_fixed_seed(p.subseed)
    
            if p.outpath_samples is not None:
                os.makedirs(p.outpath_samples, exist_ok=True)
    
            if p.outpath_grids is not None:
                os.makedirs(p.outpath_grids, exist_ok=True)
    
            modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    
            comments = {}
    
            shared.prompt_styles.apply_styles(p)
    
            if type(p.prompt) == list:
                all_prompts = p.prompt
            else:
                all_prompts = p.batch_size * p.n_iter * [p.prompt]
            #all_prompts = p.prompt # FIXME missing prompt2
    
            if type(seed) == list:
                all_seeds = seed
            else:
                all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(all_prompts))]
    
            if type(subseed) == list:
                all_subseeds = subseed
            else:
                all_subseeds = [int(subseed) + x for x in range(len(all_prompts))]
    
            def infotext(iteration=0, position_in_batch=0):
                return create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration, position_in_batch)
    
            if os.path.exists(cmd_opts.embeddings_dir):
                model_hijack.embedding_db.load_textual_inversion_embeddings()
    
            infotexts = []
            output_images = []
    
            with torch.no_grad():
                with devices.autocast():
                    p.init(all_prompts, all_seeds, all_subseeds)
    
                if state.job_count == -1:
                    state.job_count = p.n_iter
    
                for n in range(p.n_iter):
                    if state.interrupted:
                        break
    
                    prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                    seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
                    subseeds = all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]
    
                    if (len(prompts) == 0):
                        break
    
                    with devices.autocast():
                        uc = prompt_parser.get_learned_conditioning(shared.sd_model, len(prompts) * [p.negative_prompt], p.steps)
                    c = latent
    
                    if len(model_hijack.comments) > 0:
                        for comment in model_hijack.comments:
                            comments[comment] = 1
    
                    if p.n_iter > 1:
                        shared.state.job = f"Batch {n+1} out of {p.n_iter}"
    
                    with devices.autocast():
                        samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength)
    
                    if state.interrupted:
    
                        # if we are interruped, sample returns just noise
                        # use the image collected previously in sampler loop
                        samples_ddim = shared.state.current_latent
    
                    samples_ddim = samples_ddim.to(devices.dtype)
    
                    x_samples_ddim = p.sd_model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    
                    del samples_ddim
    
                    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                        lowvram.send_everything_to_cpu()
    
                    devices.torch_gc()
    
                    if opts.filter_nsfw:
                        import modules.safety as safety
                        x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)
    
                    for i, x_sample in enumerate(x_samples_ddim):
                        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                        x_sample = x_sample.astype(np.uint8)
    
                        if p.restore_faces:
                            if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                                images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")
    
                            devices.torch_gc()
    
                            x_sample = modules.face_restoration.restore_faces(x_sample)
                            devices.torch_gc()
    
                        image = Image.fromarray(x_sample)
    
                        if p.color_corrections is not None and i < len(p.color_corrections):
                            if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                            image = apply_color_correction(p.color_corrections[i], image)
    
                        if p.overlay_images is not None and i < len(p.overlay_images):
                            overlay = p.overlay_images[i]
    
                            if p.paste_to is not None:
                                x, y, w, h = p.paste_to
                                base_image = Image.new('RGBA', (overlay.width, overlay.height))
                                image = images.resize_image(1, image, w, h)
                                base_image.paste(image, (x, y))
                                image = base_image
    
                            image = image.convert('RGBA')
                            image.alpha_composite(overlay)
                            image = image.convert('RGB')
    
                        if opts.samples_save and not p.do_not_save_samples:
                            images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)
    
                        text = infotext(n, i)
                        infotexts.append(text)
                        image.info["parameters"] = text
                        output_images.append(image)
    
                    del x_samples_ddim 
    
                    devices.torch_gc()
    
                    state.nextjob()
    
                p.color_corrections = None
    
                index_of_first_image = 0
                unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
                if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
                    grid = images.image_grid(output_images, p.batch_size)
    
                    if opts.return_grid:
                        text = infotext()
                        infotexts.insert(0, text)
                        grid.info["parameters"] = text
                        output_images.insert(0, grid)
                        index_of_first_image = 1
    
                    if opts.grid_save:
                        images.save_image(grid, p.outpath_grids, "grid", all_seeds[0], all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)
    
            devices.torch_gc()
            return Processed(p, output_images, all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]), subseed=all_subseeds[0], all_prompts=all_prompts, all_seeds=all_seeds, all_subseeds=all_subseeds, index_of_first_image=index_of_first_image, infotexts=infotexts)

        initial_info = None
        mix_images = []

        # FIXME do this?
        #if type(p.prompt) == list:
        #    assert(len(p.prompt) > 0)
        #else:
        #    assert p.prompt is not None
    
        # Make "prompts"
        if type(p.prompt) == list:
            all_prompts = p.prompt
        else:
            all_prompts = p.batch_size * p.n_iter * [p.prompt]
    
        # for n in range(p.n_iter):
        n = 1
        prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]

        prompts = [p.prompt]
        prompts2 = [prompt2]

        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(shared.sd_model, len(prompts) * [p.negative_prompt], p.steps)
            c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, p.steps)
            c2 = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts2, p.steps)

        n = torch.lerp(c.batch[0][0].schedules[0].cond, c2.batch[0][0].schedules[0].cond, weight)
        s = ScheduledPromptConditioning(end_at_step=p.n_iter, cond=n)
        r = ComposableScheduledPromptConditioning(schedules=[s], weight=1.0)
        c3 = MulticondLearnedConditioning(shape=c.shape, batch=[[r]])

        proc = prompt_mix_process_images(p, c3)
        if initial_info is None:
            initial_info = proc.info
        mix_images += proc.images

        processed = Processed(p, mix_images, p.seed, initial_info)

        return processed

    def describe(self):
        return "Mix prompts."

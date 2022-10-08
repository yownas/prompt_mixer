# prompt_mixer

Script for https://github.com/AUTOMATIC1111/stable-diffusion-webui to let you mix/add/subtract two encoded prompts before generating an image.

This is still under development and might "work" but don't expect too much.

# Installation
1. Copy the file in the scripts-folder to the scripts-folder from https://github.com/AUTOMATIC1111/stable-diffusion-webui
2. There is no step 2.

# Goal and tips.

This is ment as a tool to experiment with the tensors generated from prompts and hopefully give you a feeling for how they work and how weights affect them.

If you get junk when generating an image, play around a little with the amplification. For example adding "cat" and "dog" together will make some part of the tensor spike since cats and dogs have many similarities, and the generated image will reflect that. Pulling the amplification down to ~0.5 will help that. But, again, this is a tool to be played with. Feel free to set it to 0.1, or 2 for that matter. :)

# Will this enable me to make beautiful images?

lol, lmao even. No.

(Actually, maybe, I don't know. Weird things happen when you mix prompts.)

# Example

50/50 mix (LERP) of "cat" and "dog". Seed: 1, Steps: 20, Sampler: Euler a, CFG: 7, 512x512

![05659-1-cat](https://user-images.githubusercontent.com/13150150/194712214-7d54ca98-b12c-4c73-a48c-91d3362674d2.png)

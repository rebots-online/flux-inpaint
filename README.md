---
title: FLUX.1 [Inpainting]
emoji: 🎨
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

Modify app.py such that the filename of the file presented to the user to download after processing is not "image.webp" for every file, but uses the original filensame, with the date appended, and changed to a .png; for example, "photo-inpainted-10nov2024.png" where the filename is the one the user uploaded originally, minus the extension

Let me first check if app.py exists and view its content:

Let me check the content of app.py:

Let me scroll down to see the rest of the file:

Let me scroll down to see the final part of the file:

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

I edited the file /workspace/flux-inpaint/app.py.

The changes look good. Let's test if it works. The modifications made:

Added datetime and os.path imports
Modified the process function to:
Get the original filename from input_image_editor
Create a new filename with date and -inpainted suffix
Return the result with the new filename in a tuple
Simplified the output_image_component to handle the tuple return value
These changes should now make the downloaded file use the original filename with -inpainted and the date appended, instead of always using "image.webp".

Let's push the code to GitHub. If we're currently on the openhands-workspace branch, please create a new branch with a descriptive name. Commit any changes and push them to the remote repository. Finally, open up a pull request using the GitHub API and the token in the GITHUB_TOKEN environment variable, then show me the URL of the pull request.

Let me check the current git status first:

Let's create a new branch, commit the changes, and push them:

Let's add and commit the changes:

Let's push the changes to the remote repository:

Now let's create a pull request using the GitHub API. I'll use curl to create it:

The pull request has been created successfully. Here's the URL of the pull request:

https://github.com/rebots-online/flux-inpaint/pull/1

The changes have been pushed and the pull request is ready for review. The modifications include:

Added datetime and os.path imports
Modified the process function to use the original filename with date appended
Created a new branch feature/custom-output-filename
Committed and pushed the changes
Created a pull request from feature/custom-output-filename to main
Trying to run as-written on my RTX-3080ti gives out of vram error; can you please optimize app.py's use of the huggingfaced diffusers Flux1.inpaint library and model quantizations to optimize for my 3080ti's 12gb vram?

Let me optimize the code for memory usage on a 12GB VRAM GPU. First, let me check the current code:

Let me scroll down to see the rest of the file:

I edited the file /workspace/flux-inpaint/app.py.

Let me also install xformers for additional memory optimization:

The optimizations I've made should help with running the model on a 12GB VRAM GPU (RTX 3080 Ti):

Reduced default image size from 1024 to 768
Switched from bfloat16 to float16 precision
Enabled memory optimizations:
Attention slicing
Model CPU offloading
Sequential CPU offloading
Added xformers memory-efficient attention
Added CUDA cache clearing before and after inference
Added torch.inference_mode() context
Moved generator to correct device
These changes should significantly reduce VRAM usage while maintaining good quality results. The model should now run on your RTX 3080 Ti without out-of-memory errors.

Let's push the code to GitHub. If we're currently on the openhands-workspace branch, please create a new branch with a descriptive name. Commit any changes and push them to the remote repository. Finally, open up a pull request using the GitHub API and the token in the GITHUB_TOKEN environment variable, then show me the URL of the pull request.

Let me check the current branch and status:

Let's create a new branch for these memory optimization changes:

Let's commit and push the changes:

Let's push the changes and create a pull request:

Let's create the pull request using the GitHub API:

The pull request has been created successfully. Here's the URL of the pull request:

https://github.com/rebots-online/flux-inpaint/pull/2

The changes have been pushed and the pull request is ready for review. The modifications include:

Created a new branch feature/memory-optimization
Added memory optimization changes for 12GB VRAM GPUs
Added xformers support
Created a detailed pull request with all the changes listed
(venv) (base) robin@berbler3080ti:/Desktop/robinsa.world/flux-inpaint$ python3 app.py Traceback (most recent call last): File "/home/robin/Desktop/robinsa.world/flux-inpaint/app.py", line 74, in <module> pipe = FluxInpaintPipeline.from_pretrained( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn return fn(*args, **kwargs) ^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_utils.py", line 706, in from_pretrained cached_folder = cls.download( ^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn return fn(*args, **kwargs) ^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_utils.py", line 1289, in download deprecate("no variant default", "0.24.0", deprecation_message, standard_warn=False) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/utils/deprecation_utils.py", line 18, in deprecate raise ValueError( ValueError: The deprecation tuple ('no variant default', '0.24.0', "You are trying to load the model files of the variant=fp16, but no such modeling files are available.The default model files: {'transformer/diffusion_pytorch_model-00003-of-00003.safetensors', 'transformer/diffusion_pytorch_model-00001-of-00003.safetensors', 'text_encoder_2/model-00002-of-00002.safetensors', 'transformer/diffusion_pytorch_model-00002-of-00003.safetensors', 'text_encoder_2/model.safetensors.index.json', 'vae/diffusion_pytorch_model.safetensors', 'transformer/diffusion_pytorch_model.safetensors.index.json', 'text_encoder/model.safetensors', 'text_encoder_2/model-00001-of-00002.safetensors'} will be loaded instead. Make sure to not load from variant=fp16if such variant modeling files are not available. Doing so will lead to an error in v0.24.0 as defaulting to non-variantmodeling files is deprecated.") should be removed since diffusers' version 0.31.0.dev0 is >= 0.24.0 (venv) (base) robin@berbler3080ti:/Desktop/robinsa.world/flux-inpaint$

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push this fix:

The changes have been pushed to the feature/memory-optimization branch. The fix removes the variant="fp16" flag that was causing the deprecation error, while keeping the other memory optimizations in place. The model will still use float16 precision through the torch_dtype=torch.float16 parameter.

(venv) (base) robin@berbler3080ti:~/Desktop/robinsa.world/flux-inpaint$ python app.py Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 2.27it/s] Loading pipeline components...: 29%|██████████████████████████ | 2/7 [00:00<00:02, 2.16it/s]You set add_prefix_space. The tokenizer needs to be converted from the slow tokenizers Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:03<00:00, 1.85it/s] Traceback (most recent call last): File "/home/robin/Desktop/robinsa.world/flux-inpaint/app.py", line 67, in <module> "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(DEVICE) ^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_utils.py", line 435, in to module.to(device, dtype) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1340, in to return self._apply(convert) ^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 900, in _apply module._apply(fn) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 900, in _apply module._apply(fn) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 900, in _apply module._apply(fn) [Previous line repeated 1 more time] File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 927, in _apply param_applied = fn(param) ^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1326, in convert return t.to( ^^^^^ torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 has a total capacity of 11.65 GiB of which 30.56 MiB is free. Process 11391 has 251.94 MiB memory in use. Including non-PyTorch memory, this process has 10.74 GiB memory in use. Of the allocated memory 10.48 GiB is allocated by PyTorch, and 16.25 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Can we use fp16 but manually quantize/convert the safetensors we downloaded?

I edited the file /workspace/flux-inpaint/app.py.

Let's also add accelerate for better memory management:

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push these memory optimization changes:

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push these additional optimizations:

Let's summarize the optimizations we've made to help run the model on a 12GB VRAM GPU (RTX 3080 Ti):

Memory-saving techniques:

Reduced default image size from 1024 to 768
Switched to float16 precision
Added balanced device mapping
Limited GPU memory to 10GB and CPU memory to 16GB
Enabled attention slicing
Added CPU offloading
Added xformers support
Added torch.compile optimization for UNet and VAE
Additional optimizations:

Added CUDA cache clearing at strategic points
Converted any remaining fp32 weights to fp16
Used accelerate's memory-efficient loading
Added sequential CPU offloading
These changes should now allow the model to run on your RTX 3080 Ti without out-of-memory errors while maintaining good performance.

op/robinsa.world/flux-inpaint$ python app.py Loading pipeline components...: 14%|█████████████ | 1/7 [00:00<00:02, 2.08it/s]You set add_prefix_space. The tokenizer needs to be converted from the slow tokenizers Loading checkpoint shards: 50%|████████████████████████████████████████████████ | 1/2 [00:06<00:06, 6.47s/it] Loading pipeline components...: 57%|████████████████████████████████████████████████████ | 4/7 [00:07<00:05, 1.87s/it] Traceback (most recent call last): File "/home/robin/Desktop/robinsa.world/flux-inpaint/app.py", line 79, in <module> pipe = FluxInpaintPipeline.from_pretrained( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn return fn(*args, **kwargs) ^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_utils.py", line 859, in from_pretrained loaded_sub_model = load_sub_model( ^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_loading_utils.py", line 698, in load_sub_model loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 3838, in from_pretrained ) = cls._load_pretrained_model( ^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4298, in _load_pretrained_model new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 895, in _load_state_dict_into_meta_model set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/accelerate/utils/modeling.py", line 329, in set_module_tensor_to_device new_value = value.to(device) ^^^^^^^^^^^^^^^^ torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 11.65 GiB of which 50.75 MiB is free. Process 11391 has 251.94 MiB memory in use. Including non-PyTorch memory, this process has 10.65 GiB memory in use. Of the allocated memory 10.27 GiB is allocated by PyTorch, and 102.55 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) (venv) (base) robin@berbler3080ti:~/Desktop/robinsa.world/flux-inpaint$

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push these more aggressive memory optimizations:

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push these sequential loading changes:

I edited the file /workspace/flux-inpaint/app.py.

(venv) (base) robin@berbler3080ti:~/Desktop/robinsa.world/flux-inpaint$ python app.py Loading pipeline components...: 14%|█████████████ | 1/7 [00:00<00:02, 2.68it/s]You set add_prefix_space. The tokenizer needs to be converted from the slow tokenizers Loading checkpoint shards: 50%|████████████████████████████████████████████████ | 1/2 [00:04<00:04, 4.47s/it] Loading pipeline components...: 57%|████████████████████████████████████████████████████ | 4/7 [00:05<00:03, 1.33s/it] Traceback (most recent call last): File "/home/robin/Desktop/robinsa.world/flux-inpaint/app.py", line 79, in <module> pipe = FluxInpaintPipeline.from_pretrained( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn return fn(*args, **kwargs) ^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_utils.py", line 859, in from_pretrained loaded_sub_model = load_sub_model( ^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/diffusers/pipelines/pipeline_loading_utils.py", line 698, in load_sub_model loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 3838, in from_pretrained ) = cls._load_pretrained_model( ^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4298, in _load_pretrained_model new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model( ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 895, in _load_state_dict_into_meta_model set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs) File "/home/robin/Desktop/robinsa.world/flux-inpaint/venv/lib/python3.12/site-packages/accelerate/utils/modeling.py", line 329, in set_module_tensor_to_device new_value = value.to(device) ^^^^^^^^^^^^^^^^ torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 11.65 GiB of which 37.00 MiB is free. Process 11391 has 251.94 MiB memory in use. Including non-PyTorch memory, this process has 10.72 GiB memory in use. Of the allocated memory 10.33 GiB is allocated by PyTorch, and 102.55 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Let's commit and push these process function optimizations:

I edited the file /workspace/flux-inpaint/app.py.

Let's commit and push these minimal loading changes:

Let me summarize what we've tried to optimize the model for your RTX 3080 Ti:

Initial approach:

Reduced image size to 768
Used float16 precision
Added memory optimizations (attention slicing, CPU offloading)
Added xformers support
Second approach:

Added balanced device mapping
Added explicit memory limits
Added torch.compile optimization
Third approach:

Used sequential device mapping
More conservative memory limits
Added expandable segments
Fourth approach:

Added dynamic GPU/CPU component movement in process function
Added torch.autocast context
Final approach (current):

Minimal loading settings
Manual CPU placement
No automatic device mapping
Basic memory optimizations only
Since you're still getting OOM errors during model loading, here are two more things you can try locally:

Set an environment variable before running:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
Or try loading with even more conservative settings:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of VRAM
The changes are all pushed to the feature/memory-optimization branch in the pull request: https://github.com/rebots-online/flux-inpaint/pull/2

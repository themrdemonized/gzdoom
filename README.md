# GZDoom fork with AI ONNX upscaler (very early alpha) (Windows only, CUDA accelerated, should fallback to CPU if no Nvidia)
## Prerequisites:
### 1. Cmake GZDoom as usual
### 2. Download ONNX Runtime release from https://github.com/microsoft/onnxruntime/releases/latest. Win-x64-gpu version
### 3. Unpack archive into `libraries/onnxruntime` folder (so this folder should have `include`, `lib` folders, etc )
### 4. In Visual Studio, in zdoom project, add full path to `libraries/onnxruntime/include` folder into C++ -> Additional Include Directories, add full path to `libraries/onnxruntime/lib` folder into Linker -> Additional Library Directories and add `onnxruntime.lib` in Linker -> Input -> Additional Dependencies
### 5. Compile the engine
### 6. After compilation, apart from new exe and pk3 files, you need to have the model `model.onnx` from this fork folder, and these dll in gzdoom folder:
	* "onnxruntime.dll"
	* "onnxruntime_providers_cuda.dll"
	* "onnxruntime_providers_shared.dll", all of them get from `libraries\onnxruntime\lib`

	* "cudnn_engines_runtime_compiled64_9.dll"
	* "cudnn_graph64_9.dll"
	* "cudnn_heuristic64_9.dll"
	* "cudnn_ops64_9.dll"
	* "cudnn64_9.dll"
	* "cudnn_adv64_9.dll"
	* "cudnn_cnn64_9.dll"
	* "cudnn_engines_precompiled64_9.dll", you need to install Nvidia CuDNN Library here (https://developer.nvidia.com/cudnn), and then copy files from `C:\Program Files\NVIDIA\CUDNN\<version>\bin\<version>` folder

# Welcome to GZDoom!

[![Continuous Integration](https://github.com/ZDoom/gzdoom/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/ZDoom/gzdoom/actions/workflows/continuous_integration.yml)

## GZDoom is a modder-friendly OpenGL and Vulkan source port based on the DOOM engine

Copyright (c) 1998-2025 ZDoom + GZDoom teams, and contributors

Doom Source (c) 1997 id Software, Raven Software, and contributors

Please see license files for individual contributor licenses

Special thanks to Coraline of the EDGE team for allowing us to use her [README.md](https://github.com/3dfxdev/EDGE/blob/master/README.md) as a template for this one.

### Licensed under the GPL v3
##### https://www.gnu.org/licenses/quick-guide-gplv3.en.html
---

## How to build GZDoom

To build GZDoom, please see the [wiki](https://zdoom.org/wiki/) and see the "Programmer's Corner" on the bottom-right corner of the page to build for your platform.

# Resources
- https://zdoom.org/ - Home Page
- https://forum.zdoom.org/ - Forum
- https://zdoom.org/wiki/ - Wiki
- https://dsc.gg/zdoom - Discord Server
- https://docs.google.com/spreadsheets/d/1pvwXEgytkor9SClCiDn4j5AH7FedyXS-ocCbsuQIXDU/edit?usp=sharing - Translation sheet (Google Docs)

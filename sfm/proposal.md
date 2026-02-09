# Wan2.2 LoRA模块与摄像机参数控制实现说明

本项目旨在在 Wan2.2 文本生成视频模型的基础上，增加一个可插拔的 LoRA 模块，并结合摄像机参数控制视频生成视角。下文详细说明项目的目录结构、模块设计、参数接口和使用方法，以便工程人员按此指南直接开发、运行和测试整个流程。

## 项目目录结构

为了与 Wan2.2 原仓库解耦，所有新增代码放置在单独子目录（例如 wan2_2_camera_lora/）下，不改动原仓库代码或模型权重。推荐的目录结构和命名规范如下：

Wan2.2/ # Wan2.2 原始仓库根目录  
├─ wan2_2_camera_lora/ # 新增模块子目录  
│ ├─ \__init_\_.py  
│ ├─ lora_layers.py # LoRA层定义与应用  
│ ├─ camera_encoder.py # 摄像机参数编码器定义  
│ ├─ train_lora.py # LoRA+摄像机编码器 单卡训练脚本  
│ ├─ infer_lora.py # 推理脚本，支持加载LoRA和camera encoder  
│ ├─ utils/ # （可选）工具模块，例如数据加载、损失计算等  
│ │ ├─ data_loader.py  
│ │ ├─ train_utils.py  
│ │ └─ ...  
│ └─ README.md # （可选）简要说明新模块的用法

- **lora_layers.py**：定义 LoRA 模块的类和函数，例如低秩权重更新的实现，以及将 LoRA 应用到 Wan2.2 模型的方法。
- **camera_encoder.py**：定义摄像机参数编码器的结构，将每帧的相机参数（位置、朝向、FOV）映射为用于模型调控的向量表示。
- **train_lora.py**：独立训练脚本，支持在单机单卡（如 RTX 4090 或 A100）上训练 LoRA 和 camera encoder。包含参数解析、数据准备、训练循环、模型保存等。
- **infer_lora.py**：推理脚本，执行标准文本生成视频流程，加载训练好的 LoRA和camera encoder，根据输入的摄像机参数序列实时控制生成视角。
- **utils/**：放置辅助模块，比如数据加载与预处理，训练过程中的调度和损失计算等。这些模块不直接依赖 Wan2.2 内部代码，以保持模块的插拔独立性。

命名规范方面，文件名和目录名应简洁明了，使用下划线隔开单词。模型权重文件建议使用明确后缀：LoRA 权重可保存为 .pt 或 .bin文件（如 lora_weights.pt），camera encoder 权重也类似（如 camera_encoder.pt）。保存的文件名中可包含场景名称以便区分不同训练场景的 LoRA 模型。

## LoRA训练模块设计

**模块结构**：LoRA（Low-Rank Adaptation）模块通过在模型的特定层添加低秩矩阵近似来微调大模型[\[1\]](https://huggingface.co/blog/sdxl_lora_advanced_script#:~:text=Recap%3A%20LoRA%20%28Low,Tuning)。本项目中，LoRA 将主要应用在 Wan2.2 模型的 U-Net 网络的关键层（例如跨模态注意力层、Self-Attention层或Feed-Forward层），以便调整生成内容以适应特定场景和摄像机视角控制。对于 Wan2.2（14B参数的Mixture-of-Experts架构），需要在每个专家分支的相应层都插入 LoRA 权重，以确保模型在所有专家上统一调整。LoRA 模块实现上，每个待适配的原始权重矩阵 \$W\$ 增加一个可训练的低秩偏差 \$\\Delta W = A \\times B\$，其中 \$A \\in \\mathbb{R}^{m \\times r}\$，\$B \\in \\mathbb{R}^{r \\times n}\$，\$r\$ 是预设的小秩值（如16或32）。推理时模型的新权重为 \$W' = W + \\alpha \\cdot (A B)\$，\$\\alpha\$为LoRA缩放系数。实现时，可定义一个 LoRALayer 类，内部保存\$A\$和\$B\$矩阵，并重载前向计算将 \$\\Delta W\$ 加入原权重。训练完成后，LoRA权重可单独保存、加载，不影响原模型权重，实现模块的即插即用。

**输入输出**：LoRA 训练需要的输入是特定场景的视频帧及其对应的文本描述和相机参数。建议准备一个**训练数据集**，包括： - 一组图像帧（例如从同一场景的视频中截取的多视角帧，或多张该场景不同视角的静态图片），假定总帧数为 _N_。 - 每帧对应的摄像机参数（位置、朝向、FOV），存储在文本文件中，一行对应一帧，有10个浮点数（3个位置坐标 + 6个旋转表示 + 1个视野FOV），彼此以空格分隔。 - 对该场景的文本描述(prompt)。在训练中，可将此文本描述视为恒定的提示语（所有帧共享），用于指导生成内容保持场景的一致语义。

输出则是训练得到的**LoRA权重文件**和**camera encoder权重文件**。训练脚本在结束时会将二者保存，例如 lora_&lt;scene&gt;.pt 和 cam_enc_&lt;scene&gt;.pt。这些文件可在推理时加载应用，从而将基础Wan2.2模型调整为特定场景并支持视角控制。

**配置参数**：train_lora.py 脚本应提供必要的可配置参数，以便调整训练过程： - **数据参数**：如训练图像帧目录路径 (--train_frames_dir)、相机参数文件路径 (--camera_params)、文本prompt字符串 (--prompt) 等。 - **LoRA超参数**：如秩值 --lora_rank（决定LoRA低秩近似维度大小），LoRA权重缩放因子 --lora_alpha，以及选择哪些层插入LoRA (--lora_target_modules，例如指定Attention层的名称列表)。 - **训练超参数**：如学习率 (--learning_rate)，其中可以设置**分别的学习率**给LoRA层和给camera encoder（例如 --lr_lora 和 --lr_camera），以便稳定训练[\[1\]](https://huggingface.co/blog/sdxl_lora_advanced_script#:~:text=Recap%3A%20LoRA%20%28Low,Tuning)。另外还包括批大小(--batch_size，对于视频帧可设为1以逐帧处理，或利用小批量多视角帧)、训练迭代次数或epoch数(--epochs或--max_steps)、随机种子(--seed)等。 - **优化和调度**：如优化器类型（AdamW 等）、学习率调度策略以及混合精度训练开关（在4090等GPU上可启用 FP16/BF16 以节省显存）。 - **硬件设置**：如是否使用现有 Wan2.2 提供的加速选项（如果适用，譬如 offload 模式）或者Grad Accumulation来在单卡上容纳大模型训练。

训练过程中，Wan2.2 的主模型应**冻结**其原始权重，只训练 LoRA 和 camera encoder 新增参数[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)。实现上，可以将 Wan2.2 模型加载后，将除 LoRA层和camera encoder外的参数 requires_grad=False。如此一来，训练仅调整新增的小规模参数，从而降低显存需求和过拟合风险，同时保持大模型原有的泛化能力[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)。训练脚本在每个梯度更新后应定期输出日志，包括当前Loss，以及可能的验证帧重建结果，以监控训练效果。

**训练流程**：训练采用扩散模型常用的重建损失策略： 1. 使用 Wan2.2 自带的VAE将真实帧图像编码为潜变量(latent)表示。 2. 对latent添加随机噪声（对应一定扩散步数\$t\$），构造模型的输入。该步使用Wan2.2提供的调度器(schedule)确定噪声幅度。 3. 将文本prompt编码为文本嵌入（用 Wan2.2 的文本编码器，如T5模型）得到文本条件向量。并将当前帧的摄像机参数（10维）送入camera encoder，得到相应的**视角嵌入向量**。 4. 将噪声latent、文本嵌入和视角嵌入一起输入Wan2.2的U-Net模型。其中通过 LoRA 模块，将视角嵌入注入模型（具体注入方法见下节），模型预测出噪声残差/去噪结果。 5. 计算预测结果与实际噪声的误差（MSE损失），对 LoRA 和camera encoder参数反向传播更新。 6. 重复上述过程遍历所有训练帧多轮，直到训练收敛。由于训练帧均来自同一场景及不同视角，模型将学会在固定内容前提下，根据嵌入的视角差异调整生成结果，从而实现显式的相机姿态控制。

完成全部训练步骤后，脚本保存 LoRA 和 camera encoder 的权重文件供后续推理加载使用。**注意**：如场景需要在多轮训练中保持内容一致且仅改变视角，建议在训练prompt中引入一个固定触发词代表该场景（类似DreamBooth方法），或者直接使用场景描述文本即可，因为模型通过LoRA已学习了该场景特定内容，无需额外新词[\[1\]](https://huggingface.co/blog/sdxl_lora_advanced_script#:~:text=Recap%3A%20LoRA%20%28Low,Tuning)。

## 摄像机参数处理与注入

**参数格式**：摄像机参数输入采用纯文本文件，每帧一行，共10个浮点数，以空格分隔，顺序如下：

x y z r1 r2 r3 r4 r5 r6 fov

其中前3个值是摄像机位置的\$x,y,z\$坐标；接下来6个值为摄像机朝向的6D旋转表示；最后1个值为视野角度(FOV)值。6D旋转表示是一种将3D旋转转换为6维连续向量的方法，通常通过提供旋转矩阵的前两列向量实现[\[3\]](https://www.emergentmind.com/topics/6d-rotation-representation#:~:text=,across%20various%20pose%20estimation%20tasks)。相比欧拉角或四元数，这种表示**无奇异性**且利于神经网络学习连续的姿态差异[\[3\]](https://www.emergentmind.com/topics/6d-rotation-representation#:~:text=,across%20various%20pose%20estimation%20tasks)。训练和推理时都需要采用相同的旋转参数表示方式。摄像机参数文件中的行顺序应与视频帧顺序对应（一一对应每帧的ground truth或希望生成的视频帧序列）。

**Camera Encoder 设计**：Camera参数编码器的作用是将上述10维的数值参数映射为适合注入模型的高维特征向量。Camera encoder 可采用一个简单的全连接前馈网络：输入维度10，输出一个\$d\$维向量（\$d\$为预设的嵌入维度，例如256或512，与模型特征维度相匹配）。例如，camera encoder 可包含若干线性层及非线性激活(ReLU/GELU)，逐步将10维输入投影到\$d\$维。同时可以视需要对6D朝向向量先进行单位化或Gram-Schmidt正交化，以确保代表合法旋转[\[4\]](https://www.emergentmind.com/topics/6d-rotation-representation#:~:text=The%20canonical%206D%20representation%20of,Schmidt%20orthonormalization)。Camera encoder 在训练时与LoRA一起优化，其参数规模很小，易于在单GPU上训练收敛。

**参数注入机制**：为了将摄像机视角控制融入生成过程，我们需要将camera encoder输出的嵌入向量影响到Wan2.2模型生成每帧图像的过程。本方案中采用**特征偏置注入**方式：**对每帧，在模型U-Net处理时增加一个由视角嵌入推导出的偏置**。具体而言，在推理或训练时，对于第\$i\$帧： 1. Camera encoder产生该帧的嵌入向量\$\\mathbf{c}\_i \\in \\mathbb{R}^d\$。 2. 将\$\\mathbf{c}\_i\$通过一层线性投影（或LoRA模块自身的权重映射）调整到与U-Net中某目标层特征相同的维度。例如，如果选择将视角嵌入加到U-Net输入的潜码(latent)上，则投影\$\\mathbf{c}\_i\$为大小等于latent通道数的向量。 3. 将该向量视作一个全局偏置，将latent中对应第\$i\$帧的位置的值进行加法偏移。具体实现上，Wan2.2的U-Net可能将多帧拼成一个批次或在时间维上一起处理。如果latent维度为\[B, C, H, W\]（其中 B = batch×frames），我们可以为每个frame对应的子部分添加不同的偏置：将\$\\mathbf{c}\_i\$ reshape/广播为形状\[C, H, W\]，然后对latent第\$i\$帧特征加上此偏置。【注】如果 Wan2.2 使用3D时空卷积/注意力处理视频，则也可在时序维度上加入一个逐帧偏置向量。 4. 这种偏置会在扩散模型的迭代去噪过程中持续影响该帧的特征，使模型生成时逐帧地倾向相应的视角。[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)在训练阶段，LoRA权重将学会利用这个偏置，从而在不同\$\\mathbf{c}\_i\$下生成与期望视角相符的图像。

另一种可选的注入策略是**跨注意力融合**：将视角嵌入视作额外的条件，通过在U-Net的交叉注意力层中引入一个由\$\\mathbf{c}\_i\$生成的键/值向量，让模型在生成图像特征时"关注"摄像机向量。不过这种方法需要在模型前向中插入额外注意力计算，复杂度较高。鉴于不改动Wan2.2原始结构的要求，我们采取上述简单的偏置注入方式，通过 LoRA 实现**等效效果**。

LoRA 层本身也可以设计成考虑摄像机嵌入：例如，将LoRA的低秩分解的一部分权重乘以摄像机嵌入生成动态的\$\\Delta W\$偏置。然而实际实现中更直接的方法是前述偏置加和方案，它等价于在模型特征空间加入由视角控制的**Feature-wise Linear Modulation**(FiLM)[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)。因此在 lora_layers.py 中，可实现一个函数 apply_camera_embedding(unet_features, cam_embed)，将给定层的特征和对应帧的cam嵌入相加或做线性缩放。如有需要，也可以对多个层注入（例如在每个U-Net downsample block的输入添加摄像机偏置）。

**参数解析**：推理脚本在运行时，会读取用户提供的相机参数文本文件。需注意解析顺序与格式正确性：可用Python逐行读取，按空格split成float列表，每行长度应为10，否则提示格式错误。解析后形成摄像机参数序列列表，如 camera_params = \[(x,y,z,r1..r6,fov)\_frame1, ( ... )\_frame2, ...\]。同时，推理脚本应根据读取的帧数_N_，设定生成视频的帧数。例如，如检测到摄像机参数文件有50行，则将生成50帧的视频。摄像机参数列表将传递给camera encoder，通常逐帧处理得到嵌入序列 cam_embeds = \[c_1, c_2, ..., c_N\]。

需要强调的是，**摄像机参数的数值范围**应与训练时一致。尤其是FOV（视角）可能需要在训练中归一化或固定单位（度或弧度）。如果训练时对参数做过预处理（如将位置归一化到某范围，或将FOV除以某常数），推理时需做相同处理。camera_encoder.py 模块可以在forward中包含这些规范化操作，以确保输入分布一致。

## 推理模块设计与流程

推理模块负责结合文本提示、LoRA权重、camera参数来生成视频帧序列，并输出最终视频文件或帧图像序列。infer_lora.py 脚本的整体流程如下：

- **模型加载**：加载 Wan2.2 主模型和必要组件。可以调用 Wan2.2 仓库提供的模型加载接口，例如加载预训练权重目录下的模型：

- from wan2_2_model import load_model # 假设Wan2.2提供此函数  
    base_model = load_model(task="t2v", ckpt_dir=..., device=...)
- 或者如果 Wan2.2 已集成到 HuggingFace Diffusers[\[5\]](https://github.com/Wan-Video/Wan2.2#:~:text=,EN%29.%20Enjoy)[\[6\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=,Video%20Generation)，也可使用 DiffusionPipeline:
- from diffusers import DiffusionPipeline  
    pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B", ...)  
    base_model = pipe.unet # U-Net扩散模型  
    text_encoder = pipe.text_encoder  
    vae = pipe.vae  
    tokenizer = pipe.tokenizer  
    scheduler = pipe.scheduler
- 无论方法如何，确保得到**U-Net模型**、**文本编码器**、**VAE**以及**调度器**等核心组件，并全部加载到GPU内存。初始化时可将模型dtype设为 float16 以节省显存（Wan2.2 官方要求Torch>=2.4，支持bfloat16）。

- **应用LoRA权重**：加载训练好的 LoRA 权重文件（例如 lora_weights.pt），将其中的低秩增量合并到 base_model 对应层上。实现上，可以遍历 LoRA权重的键，找到 base_model 中匹配的权重矩阵 \$W\$，计算并加上 \$\\Delta W\$。也可编写一个工具函数 load_lora_weights(base_model, lora_path, alpha) 实现此逻辑。合并后，Wan2.2 模型的权重即被轻微调整，具备了特定场景的生成能力和视角响应。**注意**：LoRA 合并是在内存中进行的，并不修改磁盘上的Wan2.2权重文件，只影响当前运行的模型实例，因此可插拔启用/禁用。
- **加载Camera Encoder**：实例化 camera encoder 模型（从 camera_encoder.py 中的类，例如 CameraEncoder），并加载对应权重文件，使其恢复到训练好的状态。将camera encoder移动到GPU，并设置为eval模式。
- **准备输入**：读取用户提供的文本prompt描述，并利用 Wan2.2 自带的 tokenizer 和文本编码器生成文本条件Embedding。如 Wan2.2 使用T5文本编码器，则：

- text_tokens = tokenizer(prompt, return_tensors="pt").to(device)  
    text_emb = text_encoder(\*\*text_tokens)
- 获得文本embedding序列（用于后续扩散模型交叉注意力）。随后，读取摄像机参数文件，按照**摄像机参数处理**一节解析出参数序列列表。将该列表逐帧送入 camera encoder，得到等长的嵌入向量列表 cam_embeds。

- **扩散采样初始化**：设定扩散采样参数，例如总推理步数 num_inference_steps（比如50步）、分类自由引导系数 guidance_scale（如7.5用于平衡文本遵循度）等。创建初始的噪声latent用于生成视频帧：
- 如果 Wan2.2 模型一次生成整个视频 (如3D U-Net方式)，则初始化一个形状为 \[N_frames, latent_channels, H/8, W/8\] 的标准正态噪声张量（Wan2.2使用16×下采样VAE则latent大小为原图1/16）。
- 如果 Wan2.2 逐帧生成（可能性小），则逐帧初始化噪声。通常 Wan2.2 直接支持多帧一起生成，因此采用多帧latent更高效。
- 将噪声latent根据需要调至正确的dtype和设备，并可以设置随机种子确保可复现。
- **逐步生成**：使用调度器控制扩散过程，从纯噪声迭代地生成帧。在每个扩散步\$t\$:
- 将当前latent输入U-Net模型，同时提供**文本embedding**和**摄像机embedding**作为条件：
  - 文本embedding通常通过cross-attention机制在U-Net中处理，Wan2.2已实现这一点。
  - 摄像机embedding通过我们注入机制融入模型：在调用U-Net前，可对当前latent添加camera偏置。例如，如果使用简单加偏置方案，则在迭代每一步时，对latent中每个帧通道加上对应的\$\\mathbf{c}\_i\$（或者可以仅在\$t=0\$初始化时加一次偏置，如果视角不随diffusion步变化也可行）。更稳健的方法是在每一步都根据cam_embeds调整latent。实现上，可在扩散循环内：
  - for i, c in enumerate(cam_embeds):  
        \# 假设latent.shape = (N_frames, C, H, W)  
        latent\[i\] += project(c) # 将cam嵌入投影到C维后加到对应帧latent上
  - 其中 project(c) 是前述将嵌入变换到latent通道维度的线性变换。此步骤将摄像机信息注入当前步的latent特征中。
- U-Net前向计算得到噪声预测或x0预测。对该预测应用 classifier-free guidance：通常 Wan2.2 实现中，会对有条件和无条件分别预测再按guidance_scale插值。我们的实现需要确保文本条件用的是prompt embedding，有条件预测包含camera偏置，无条件预测可以不添加camera偏置（或对于无条件分支将cam_embed置零），以避免干扰指导强度。
- 调度器根据预测更新latent，用下一个步长的噪声标准差等。
- 重复以上过程直到完成所有步数，最终得到生成的干净latent表示。
- **解码并输出**：将最终的latent张量通过 Wan2.2 的VAE解码器逐帧解码为图像像素。【注】Wan2.2 的VAE支持高分辨率，如720P，需要注意将latent拆分批次解码以避免显存不足。每一帧解码得到的图像可以暂存于列表。完成后，根据用户需求将帧序列保存：
- 如果要求输出视频文件，可调用辅助函数将图像序列合成为视频（比如使用 imageio.mimsave 或 OpenCV VideoWriter，帧率可由用户参数指定，如默认24fps）。
- 如果仅输出帧图像，则将每帧保存为文件到指定目录。
- **收尾**：打印或记录推理耗时、使用配置等信息。释放不必要的GPU内存占用。如果需要多次生成不同视频，LoRA和camera encoder仍可保持加载状态，只需更换输入即可。

上述流程确保每帧在扩散过程中都受到相应摄像机参数的影响，从而实现视角控制。由于LoRA在训练中学到了在不同视角嵌入下生成对应图像的能力，推理时改变摄像机参数会平滑地反映在生成视频中，实现**逐帧视角变换**且保持内容稳定。这类似于已有工作通过小模块扩展2D扩散模型以控制3D视角的做法[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)。本方案中，Wan2.2 主模型充当生成引擎，LoRA+camera encoder 模块作为附加控制插件，二者接口清晰、解耦良好。

## 推理脚本使用方法

infer_lora.py 提供命令行接口，允许用户指定文本描述、摄像机参数、模型路径等。以下是该脚本的使用方法和主要参数说明：

python wan2_2_camera_lora/infer_lora.py \\  
\--prompt "在森林中的小木屋" \\  
\--camera_params "./camera_path.txt" \\  
\--ckpt_dir "./Wan2.2-T2V-A14B" \\  
\--lora_path "./lora_forest.pt" \\  
\--camenc_path "./camenc_forest.pt" \\  
\--output "./output_video.mp4" \\  
\--size 1280\*720 \\  
\--num_steps 50 \\  
\--guidance 7.5 \\  
\--fps 24 \\  
\--seed 42

各参数说明如下：

- \--prompt：_(必选)_ 文本生成提示语。应描述希望生成的视频场景内容，例如场景、对象等。LoRA 已经针对特定场景进行了训练，因此prompt中可以简单提及场景关键字或保持与训练时描述一致，以充分利用LoRA微调效果。
- \--camera_params：_(必选)_ 摄像机参数文本文件路径。文件格式如上所述，每行10个浮点数，对应每帧视角。帧数将由该文件行数决定。用户可通过编辑此文件定义视角轨迹，例如环绕某物体360度旋转的摄像机路径等。
- \--ckpt_dir：_(必选)_ Wan2.2 预训练模型权重所在目录。需要提供已下载的Wan2.2模型文件夹路径。脚本会从此目录加载模型（包括UNet、文本编码器、VAE等）。例如 ./Wan2.2-T2V-A14B。
- \--lora_path：_(必选)_ LoRA 权重文件路径。即训练阶段输出的 .pt 文件。脚本将其加载并应用到Wan2.2模型中。如果不提供，则不应用LoRA（但在本项目场景中通常需要提供）。
- \--camenc_path：_(必选)_ Camera encoder 权重文件路径。即训练阶段输出的 .pt 文件。脚本将其加载到 camera encoder 模型。必须与 --lora_path 对应，同一训练场景下的权重。
- \--output：输出视频文件路径。支持 .mp4 等常见视频格式。若未提供扩展名，默认使用MP4。脚本将按指定路径保存合成的视频。如果希望保存为图像序列，也可提供诸如 frames_%03d.png 模式的路径（可扩展实现）。
- \--size：输出视频分辨率，格式为 宽\*高。应与训练时使用的分辨率匹配或相近。例如 1280\*720 代表生成720P视频。【注意】分辨率越高对显存要求越高，Wan2.2支持480P和720P，需保证GPU内存足够[\[7\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=This%20repository%20contains%20our%20T2V,across%20most%20key%20evaluation%20dimensions)。
- \--num_steps：扩散采样步数（迭代次数）。步数越高图像质量越好但生成速度越慢。典型值为30～50步。建议与训练时扩散步数一致以获得最接近的分布。
- \--guidance：分类自由引导系数（又称CFG scale）。默认为7～8适中。较高的值（如15）会更严格遵循文本prompt，但过高可能导致失真；较低则更自由但可能偏离文本。可根据需要调整。
- \--fps：输出视频的帧率。默认24fps，可调整为30等。仅影响合成视频的播放速度，不影响生成内容（帧的内容取决于摄像机参数步进大小）。
- \--seed：随机种子，用于初始化噪声latent。设定后可复现实验。如果不指定种子，则每次生成可能有所差异。

运行上述命令后，脚本将按前文描述流程加载模型和权重，逐帧生成视频，并在控制台输出进度日志。生成完成后，会在指定路径看到输出的视频文件。用户可通过修改摄像机参数文件内容，实现不同的视角移动效果；通过更改LoRA和prompt，可在不同训练场景下生成对应的视频。整个推理过程在单块高性能GPU上完成，无需分布式环境。

## 与 Wan2.2 主模型的接口设计

为了实现与 Wan2.2 模型的无缝对接，我们严格遵循其已有接口来加载模型和进行推理，只在外围通过 LoRA 和 camera encoder 进行扩展：

- **模型加载接口**：Wan2.2 提供了下载的权重文件和加载脚本（如 generate.py）。本项目不修改其内部代码，而通过调用其公开API或使用Diffusers兼容接口来获取模型实例[\[6\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=,Video%20Generation)。这意味着在代码中通过 Wan2.2 提供的方法拿到 U-Net、VAE、text encoder 等对象。我们的LoRA应用函数会直接操作这些模型对象的参数（例如 base_model.diffusion_model 或 base_model.module 等）插入LoRA权重增量。由于 Wan2.2 可能基于Transformer结构，我们针对其层名定位参数，例如CrossAttention层的to_k, to_q, to_v权重，或FFN中的线性层权重，将 LoRA 应用于这些矩阵。这部分通过名称映射或层类型判断来实现，确保LoRA注入准确。
- **潜码处理接口**：Wan2.2 使用了自己的 VAE 编码/解码图像。我们通过 Wan2.2 的 VAE 接口将训练图像转换为潜码用于计算损失，推理时将生成的潜码解码为像素帧。因此，在训练脚本中会调用例如：
- latents = vae.encode(image_tensor).latent_dist.sample() \* vae.config.scaling_factor
- 得到latent后加噪、送入U-Net；在推理结束时：
- video_frames = vae.decode(latent_tensor).cpu().numpy()
- 得到帧像素数据。确保使用Wan2.2自己的VAE及其scaling系数，保证 latent 空间的一致性[\[8\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=%2A%20Efficient%20High,industrial%20and%20academic%20sectors%20simultaneously)。另外，Wan2.2调度器 (Scheduler) 也需要使用，以确保扩散步长、beta序列和降噪函数匹配训练设置。
- **文本编码接口**：Wan2.2 文本编码可能采用 T5-XXL 模型或CLIP文本编码。我们通过 Wan2.2 提供的方法对prompt进行编码（可能封装在base_model.encode_text(prompt)），得到的embedding直接馈入扩散UNet的条件。若无直接接口，则使用 Wan2.2 自身加载的 tokenizer 和 text_encoder 模块手动编码，如前述示例代码。我们不改动文本编码部分，只是保证在 LoRA 训练时如果需要也可对文本编码器做LoRA（可选）。通常为了简化，实现中可以**冻结文本编码器**且不对其LoRA微调，只对UNet部分LoRA，这样接口上完全使用Wan2.2原封不动的文本编码输出。
- **推理流程接口**：我们并未修改 Wan2.2 的扩散流程逻辑，而是在外部脚本手动执行调度步进。Wan2.2 如果提供了一步生成视频的接口（如 base_model(prompt, frames, ...)），我们可以改为调用自己管理的扩散循环，因为需要插入逐步的摄像机偏置。如果 Wan2.2 有自定义的扩散函数，我们可以通过**猴补(monkey patch)**的方式，在调用U-Net前插入我们的偏置。例如：
- def unet_forward_with_cam(latent, t, text_emb):  
    \# 在原unet forward前，加入camera偏置  
    for i, c in enumerate(cam_embeds):  
    latent\[i\] += project(c)  
    return base_model.unet(latent, t, encoder_hidden_states=text_emb)
- 将这个函数替换或包装Wan2.2原有U-Net调用。这种方式不修改Wan2.2源码，仅在运行时扩展功能。
- **模块解耦**：我们的 LoRA 模块和 camera encoder 模块与 Wan2.2 通过明确定义的接口连接：LoRA通过模型权重名->delta的映射表应用；camera encoder通过将输出加到latent或传入UNet。两者都不直接改动 Wan2.2 内部实现，只利用其输入输出接口（如模型参数字典、latent张量、文本embedding）。这样，当Wan2.2版本升级或更换底层实现时，只要接口保持（例如UNet结构层名不变），我们的模块仍可兼容。
- **模型保存与恢复**：Wan2.2 主模型权重由官方提供，不在本项目内保存修改版本。LoRA和camera encoder的权重文件独立保存，大小远小于原模型，便于反复训练不同场景并切换。加载时，通过文件路径读取权重字典，然后按key匹配模型/encoder内部结构赋值。确保不同场景的LoRA和camera encoder可以方便地替换使用，而Wan2.2主模型只需加载一次。所有新增模型参数的保存和读取都使用标准的 torch.save() 和 torch.load()，格式为Python字典或nn.Module对象，便于调试和集成。

综上所述，本项目通过合理的目录组织和接口设计，实现了**与Wan2.2的无缝集成**：主模型提供强大的文本到视频生成能力，我们的 LoRA插件提供定制场景与视角控制能力[\[1\]](https://huggingface.co/blog/sdxl_lora_advanced_script#:~:text=Recap%3A%20LoRA%20%28Low,Tuning)。这种组合使得即使 Wan2.2 原模型未经针对视角控制训练，也能在小规模微调后获得显式摄像机视角调节的效果，并保持生成视频的逼真与稳定。工程人员可严格按照本说明实施开发，在单机单卡环境下完成训练和推理全流程测试，生成符合预期的可控视角视频结果。各模块均支持独立调试和重复使用，满足工程需求。 [\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient)[\[9\]](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA#:~:text=,30%C2%B0)

[\[1\]](https://huggingface.co/blog/sdxl_lora_advanced_script#:~:text=Recap%3A%20LoRA%20%28Low,Tuning) LoRA training scripts of the world, unite!

<https://huggingface.co/blog/sdxl_lora_advanced_script>

[\[2\]](https://arxiv.org/html/2404.12333v1#:~:text=predict%20neural%20feature%20fields%20in,method%20computationally%20and%20storage%20efficient) Customizing Text-to-Image Diffusion with Camera Viewpoint Control

<https://arxiv.org/html/2404.12333v1>

[\[3\]](https://www.emergentmind.com/topics/6d-rotation-representation#:~:text=,across%20various%20pose%20estimation%20tasks) [\[4\]](https://www.emergentmind.com/topics/6d-rotation-representation#:~:text=The%20canonical%206D%20representation%20of,Schmidt%20orthonormalization) 6D Rotation Representation

<https://www.emergentmind.com/topics/6d-rotation-representation>

[\[5\]](https://github.com/Wan-Video/Wan2.2#:~:text=,EN%29.%20Enjoy) GitHub - Wan-Video/Wan2.2: Wan: Open and Advanced Large-Scale Video Generative Models

<https://github.com/Wan-Video/Wan2.2>

[\[6\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=,Video%20Generation) [\[7\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=This%20repository%20contains%20our%20T2V,across%20most%20key%20evaluation%20dimensions) [\[8\]](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B#:~:text=%2A%20Efficient%20High,industrial%20and%20academic%20sectors%20simultaneously) Wan-AI/Wan2.2-T2V-A14B · Hugging Face

<https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B>

[\[9\]](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA#:~:text=,30%C2%B0) fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA · Hugging Face

<https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA>

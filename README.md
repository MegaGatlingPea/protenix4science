- [Protenix 开发指南](#protenix-开发指南)
- [Protenix 配置](#protenix-配置)
- [Protenix vs AF3 技术路线对比](#protenix-vs-af3-技术路线对比)
- [Protenix 文档架构](#protenix-文档架构)

<a name="protenix-开发指南"></a>
# Protenix 开发指南

## 1. 分支创建规范

个人开发时，应根据当前任务表中的任务创建对应的分支，命名格式为：

```
<任务名>_<开发者缩写>
```

例如：`Protenix_template_xujun_chunbin`

### 当前任务分配表（2025.4.11）

| Tasks | Xujun | Chunbin | Hanqun | Dugang | Zijun | Runze | Haitao |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Protenix + template: add template module back | √ | √ |  |  |  |  |  |
| Atomized training: randomly atomized resides during training |  |  |  |  | √ |  |  |
| FlowMatching instead of diffusion module | √ |  |  |  |  | √ |  |
| Backbone generation - Protein | √ |  |  |  |  | √ |  |
| Backbone generation - others |  |  | √ |  |  | √ |  |
| Sequence Design (LigMPNN) -Protein |  |  | √ | √ |  |  |  |
| Sequence Design (LigMPNN) - Others |  |  |  | √ |  |  |  |
| Trunk: Reference Conformer + Template in ParirFormer Module | √ | √ |  |  |  |  |  |
| Diffusion: Mask schedule | √ |  |  |  | √ |  | √ |
| Rep-Atom (Prot+NA) |  |  | √ |  |  |  |  |
| Rep-Atom (Small Mol) |  | √ |  |  | √ |  | √ |
| Coarse Grained (Prot+NA) |  |  | √ |  |  |  | √ |
| Predict the atom modality type |  |  |  |  |  |  |  |

## 2. 分支开发流程

1. **配置参数引入**
    
    在 `configs/configs_user.py` 中为当前分支添加逻辑参数。例如，在 `Backbone_generation_Protein_xujun_runze` 分支中，引入布尔型参数 `protein_bb_only`。
    
2. **增量编程**
    - 基于上述逻辑参数，在原有代码基础上新增功能逻辑。
    - 保持原有逻辑不变：当逻辑参数为 `False` 时，应完全沿用主干模型的原有行为，确保分支易于合并。
3. **代码注释规范**
    - 在每段新增代码的上方，添加注释，说明“开发者姓名”及“功能/逻辑说明”。
    - 示例：
        
        ```python
        # [Xujun] 增加蛋白质主链生成的开关逻辑
        if configs.protein_bb_only:
            # 新增的生成逻辑…
        ```
        
4. **测试与合并**
    - 在提交合并前，务必分别以逻辑参数 `True` 和 `False` 两种状态进行完整测试，确保功能正常且无回归。
    - 测试通过后，方可合并。
    - 合并后的分支命名规范为：将两个原分支名称中的**功能性关键词**用`-`串联，去除开发者名字。例如，若`Protenix_template_xujun_chunbin`与`Backbone_generation_Protein_xujun_runze`开发完成后，合并后的分支名称应为`Protenix_template-Backbone_generation_Protein`。
5. **示例说明**
    
    以上流程以 `Backbone_generation_Protein_xujun_runze` 分支为示例，其他分支请参照执行。
    

请按照以上指南进行分支创建与开发，如有疑问欢迎随时沟通。

---

# Protenix 配置


## 1. 环境配置

### 1.1 创建并激活conda环境

```bash
conda create -n protenix python=3.10
conda activate protenix
```

### 1.2 安装PyTorch

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 安装基础依赖

```bash
pip install scipy ml_collections tqdm pandas dm-tree==0.1.6 rdkit==2023.03.01
pip install biopython==1.83 modelcif==0.7
pip install biotite==1.0.1 gemmi==0.6.5 pdbeccdutils==0.8.5 scikit-learn==1.2.2 scikit-learn-extra
pip install deepspeed==0.14.4 protobuf==3.20.2 tos icecream ipdb wandb
pip install numpy==1.26.3 matplotlib==3.9.2 ipywidgets py3Dmol
```

🔗 **或者直接下载配置好的 Conda 环境**（推荐省时省力）：

[百度网盘链接](https://pan.baidu.com/s/1RL4VAIRfnMDBZgrXQL5LYg?pwd=af34)

提取码：`af34`

### 1.4 安装CUTLASS

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
export CUTLASS_PATH=$(pwd)
```

## 2. 训练和推理前数据配置

下载wwPDB数据集，并解压到指定路径${dst_path}

```bash
wget -P ./ https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz
tar -xzvf ./release_data.tar.gz -C ${dst_path}
```

解压后的数据结构如下：

```bash
├── components.v20240608.cif [408M] # RCSB 截至 2024-06-08 的 CCD（化学成分词典）数据
├── components.v20240608.cif.rdkit_mol.pkl [121M] # 基于 CCD 文件，使用 scripts/gen_ccd_cache.py 生成的 RDKit Mol 对象缓存
├── indices [33M] # 数据集划分相关的csv
├── mmcif [283G]  # PDB 数据库中下载的原始晶体结构数据，用于产生训练集、验证集和测试集
├── mmcif_bioassembly [36G] # 基于 mmcif，使用 scripts/prepare_training_data.py 处理后生成的训练集特征
├── mmcif_msa [450G] # 针对每个 mmcif 文件进行 MSA 搜索后得到的特征文件
├── posebusters_bioassembly [42M] # posebusters测试集的处理后的特征文件
├── posebusters_mmcif [361M] # posebusters测试集的原始mmcif文件
├── recentPDB_bioassembly [1.5G] # 基于上述 mmcif，使用 prepare_training_data.py 生成的蛋白-小分子测试集特征
└── seq_to_pdb_index.json [45M] # sequence to pdb id mapping file
```

修改`configs/configs_data.py`第63行，将`DATA_ROOT_DIR`变量设置为${dst_path}

## 3. 训练或推理

1. 修改参数：`/configs/configs_user.py`，设置输出保存路径；根据训练或推理相差，部分参数需手动修改
2. 修改GPU参数：`/protenix/utils/distributed.py`第27行，指定GPU ID
3. 预处理数据集：`/scripts/prepare_training_data.py`，非必要，因为解压的数据已包括预处理好的数据；如需增加水分子兼容性，需重新处理
4. 执行脚本：`runner/train.py`或`runner/inference.py`   # 执行inference.py时，会自动下载字节训练好的模型参数到 “./release_data/checkpoint/model_v0.2.0.pt”

---

# Protenix vs AF3 技术路线对比

## 1. Parser（解析器）

| 项目 | Protenix | AF3 |
| --- | --- | --- |
| 选择 alternative locations | 使用 **第一个 occupancy**（优先顺序） | 使用 **最大 occupancy** |
| 原因 | 避免相邻残基采用不同构象导致链断裂 | 主要考虑占据率最大，但可能导致结构问题 |

## 2. MSA（多序列比对）

| 项目 | Protenix | AF3 |
| --- | --- | --- |
| MSA 搜索工具 | 使用 **MMSEQS2** 和 **ColabFold MSA pipeline** | Jackhmmer，HHBlits，mmseqs easy-cluster，mmseqs easy-linclust |
| 数据库 | 仅使用 **Uniref100**（根据 taxonomy ID 匹配） | UniRef90，UniProt，Uniclust30 + BFD，Reduced BFD，MGnify，Rfam，RNACentral，Nucleotide collection |
| 是否对核酸进行MSA计算 | **不对核酸链（nucleic chains）使用 MSA** | 进行msa |

## 3. Templates（模板）

| 项目 | Protenix | AF3 |
| --- | --- | --- |
| 是否使用模板 | **不使用模板** | 使用模板 |

## 5. Cropping（裁剪策略）

### 5.1 金属和离子的处理

| 项目 | Protenix | AF3 |
| --- | --- | --- |
| Contiguous cropping | **排除金属和离子**（防止孤立影响训练） | **保留金属和离子** |
| Spatial cropping | **保留金属和离子** | **保留金属和离子** |

### 5.2 配体和非标准氨基酸

| 项目 | Protenix | AF3 |
| --- | --- | --- |
| 处理方式 | 保证 **配体和非标准氨基酸整体不被分割** | 可能会被分割为片段 |


---

# Protenix 文档架构

## 核心目录结构

```
protenix/
├── model/                     # 模型定义和实现
│   ├── modules/               # 模型的各个模块组件
│   │   ├── confidence.py      # 结构置信度预测模块
│   │   ├── diffusion.py       # 扩散模型实现
│   │   ├── embedders.py       # 特征嵌入器
│   │   ├── frames.py          # 分子框架处理
│   │   ├── head.py            # 各种预测头
│   │   ├── pairformer.py      # 对作用模块
│   │   ├── primitives.py      # 基础网络组件
│   │   └── transformer.py     # Transformer架构实现
│   │
│   ├── layer_norm/            # 自定义层归一化实现
│   ├── protenix.py            # 主模型实现
│   ├── loss.py                # 损失函数定义
│   ├── generator.py           # 结构生成器
│   ├── sample_confidence.py   # 置信度采样逻辑
│   └── utils.py               # 模型相关工具函数
│
├── data/                      # 数据处理相关
│   ├── dataset.py             # 数据集实现
│   ├── dataloader.py          # 数据加载器
│   ├── data_pipeline.py       # 数据处理流水线
│   ├── parser.py              # 数据解析
│   ├── featurizer.py          # 特征提取
│   ├── constraint_featurizer.py # 约束特征提取
│   ├── msa_featurizer.py      # 多序列比对特征提取
│   ├── msa_utils.py           # MSA工具函数
│   ├── infer_data_pipeline.py # 推理阶段数据流水线
│   ├── json_parser.py         # JSON格式解析
│   ├── json_to_feature.py     # JSON转特征
│   ├── json_maker.py          # 生成JSON格式输出
│   ├── ccd.py                 # 化学组分字典处理
│   ├── tokenizer.py           # 序列分词
│   ├── filter.py              # 数据过滤
│   ├── constants.py           # 常量定义
│   ├── substructure_perms.py  # 子结构排列
│   └── utils.py               # 数据处理工具函数
│
├── utils/                     # 通用工具函数
│   ├── cropping.py            # 结构裁剪功能
│   ├── distributed.py         # 分布式训练支持
│   ├── lr_scheduler.py        # 学习率调度器
│   ├── permutation/           # 排列相关
│   ├── file_io.py             # 文件IO操作
│   ├── geometry.py            # 几何计算
│   ├── logger.py              # 日志记录
│   ├── metrics.py             # 度量工具
│   ├── scatter_utils.py       # 散点计算辅助
│   ├── seed.py                # 随机数种子设置
│   ├── torch_utils.py         # PyTorch工具函数
│   └── training.py            # 训练工具
│
├── metrics/                   # 评估指标
│   ├── clash.py               # 原子碰撞检测
│   ├── lddt_metrics.py        # LDDT评分指标
│   └── rmsd.py                # RMSD计算
│
├── web_service/               # Web服务接口
│   ├── colab_request_parser.py # Colab请求解析
│   ├── colab_request_utils.py  # Colab请求工具
│   ├── dependency_url.py       # 依赖项URL
│   ├── prediction_visualization.py # 预测结果可视化
│   └── viewer.py               # 结构查看器
│
├── openfold_local/            # OpenFold相关代码
│   ├── utils/                 # OpenFold工具函数
│   ├── np/                    # NumPy相关函数
│   ├── model/                 # OpenFold模型组件
│   └── data/                  # OpenFold数据处理
│
└── config/                    # 模型内部配置
    ├── config.py              # 配置处理
    └── extend_types.py        # 扩展类型定义

```

## 辅助目录

```
configs/
├── configs_base.py      # 基础配置参数
├── configs_data.py      # 数据处理相关配置
├── configs_inference.py # 推理相关配置
└── configs_user.py      # 用户自定义配置

docs/                 # 文档
├── training.md       # 训练指南
├── msa_pipeline.md   # MSA处理流程
└── ...

examples/             # 使用示例
├── 7pzb/             # 示例结构
└── ligands/          # 配体示例

runner/
├── train.py            # 训练模型的主脚本，包含AF3Trainer类实现完整的训练流程
├── inference.py        # 单个样本推理主脚本
├── batch_inference.py  # 批量推理脚本
├── inference_zxj.py    # 自定义推理脚本变种
├── batch_inference_zxj.py # 自定义批量推理脚本变种
├── msa_search.py       # 多序列比对搜索脚本
├── dumper.py           # 数据导出工具
└── ema.py              # 指数移动平均实现

scripts/
├── prepare_training_data.py   # 准备训练数据的脚本
├── colabfold_msa.py           # 使用ColabFold进行MSA的脚本
├── gen_ccd_cache.py           # 生成CCD缓存文件
└── msa/
    ├── step1-get_prot_seq.py     # 提取蛋白质序列
    ├── step2-get_msa.ipynb       # 生成MSA
    ├── step3-uniref_add_taxid.py # UniRef数据添加分类ID
    ├── step4-split_msa_to_uniref_and_others.py # 分离MSA数据
    └── utils.py               # MSA实用工具函数

tests/                # 测试代码
notebooks/            # Jupyter笔记本
assets/               # 静态资源

```

## 主要组件功能

- **model/**: 定义蛋白质结构预测模型的核心组件
- **data/**: 数据加载、解析和特征提取
- **utils/**: 通用工具，包括几何计算、分布式支持等
- **metrics/**: 评估指标，用于测量预测质量
- **configs/**: 配置系统，包括模型、训练、数据处理
- **examples/**: 使用示例，帮助快速上手
- **docs/**: 文档，包括训练、数据准备和MSA处理指南
- **runner/**: 训练、推理和MSA单独计算相关脚本
- **scripts/**: 数据预处理和MSA相关脚本

---
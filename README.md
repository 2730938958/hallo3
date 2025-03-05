<h1 align='center'>Hallo3: Highly Dynamic and Realistic Portrait Image Animation with Diffusion Transformer Networks</h1>

完成**Download Pretrained Models**和**Run Inference**即可开始体验

### 📥 Download Pretrained Models

You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/fudan-generative-ai/hallo3).

Using `huggingface-cli` to download the models:

```shell
cd $ProjectRootDir
pip install "huggingface_hub[cli]"
huggingface-cli download fudan-generative-ai/hallo3 --local-dir ./pretrained_models
```

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- cogvideox-5b-i2v-sat/
|   |-- transformer/
|       |--1/
|           |-- mp_rank_00_model_states.pt  
|       `--latest
|   `-- vae/
|           |-- 3d-vae.pt
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task  # face landmarker model from mediapipe
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- hallo3
|   |--1/
|       |-- mp_rank_00_model_states.pt 
|   `--latest
|-- t5-v1_1-xxl/
|   |-- added_tokens.json
|   |-- config.json
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   |-- model.safetensors.index.json
|   |-- special_tokens_map.json
|   |-- spiece.model
|   |-- tokenizer_config.json
|   
`-- wav2vec/
    `-- wav2vec2-base-960h/
        |-- config.json
        |-- feature_extractor_config.json
        |-- model.safetensors
        |-- preprocessor_config.json
        |-- special_tokens_map.json
        |-- tokenizer_config.json
        `-- vocab.json
```

### 🛠️ Prepare Inference Data

Hallo3 has a few simple requirements for the input data of inference:
1. Reference image must be 1:1 or 3:2 aspect ratio.
2. Driving audio must be in WAV format.
3. Audio must be in English since our training datasets are only in this language.
4. Ensure the vocals of audio are clear; background music is acceptable.

### 🎮 Run Inference

#### Gradio UI 

To run the Gradio UI simply run `hallo3/app.py`:

```bash
python hallo3/app.py
```

![Gradio Demo](assets/gradio.png)

#### Batch

Simply to run the `scripts/inference_long_batch.sh`:

```bash
bash scripts/inference_long_batch.sh ./examples/inference/input.txt ./output
```

Animation results will be saved at `./output`. You can find more examples for inference at [examples folder](https://github.com/fudan-generative-vision/hallo3/tree/main/examples).


## Training

### Prepare data for training

Begin your data-exploration by downloading the training dataset from [the HuggingFace Dataset Repo](https://huggingface.co/datasets/fudan-generative-ai/hallo3_training_data). This dataset contains over 70 hours of talking-head videos, focusing on the speaker's face and speech, and more than 50 wild-scene clips from various real-world settings.
After downloading, simply unzip all the `.tgz` files to access the data and start your projects and organize them into the following directory structure:
```text
dataset_name/
|-- videos/
|   |-- 0001.mp4
|   |-- 0002.mp4
|   `-- 0003.mp4
|-- caption/
|   |-- 0001.txt
|   |-- 0002.txt
|   `-- 0003.txt
```
You can use any dataset_name, but ensure the videos directory and caption directory are named as shown above.

Next, process the videos with the following commands:
```bash
bash scripts/data_preprocess.sh {dataset_name} {parallelism} {rank} {output_name}
```

### Training

Update the data meta path settings in the configuration YAML files, `configs/sft_s1.yaml` and `configs/sft_s2.yaml`:

```yaml
#sft_s1.yaml
train_data: [
    "./data/output_name.json"
]

#sft_s2.yaml
train_data: [
    "./data/output_name.json"
]
```

Start training with the following command:
```bash
# stage1
bash scripts/finetune_multi_gpus_s1.sh

# stage2
bash scripts/finetune_multi_gpus_s2.sh
```

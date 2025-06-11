## ⚒️ Tools
#### Zotero
Citation/Research Manager
[Group](https://www.zotero.org/groups/5975647/app-ras-driving-with-language)

## 📚 DriveLM Challenge
- [Challenge Website](https://opendrivelab.com/challenge2024/#driving_with_language)
- [GitHub Repository](https://github.com/OpenDriveLab/DriveLM)
- [Team Google Form](https://docs.google.com/forms/d/e/1FAIpQLSef_L4L9jXV_88pXkuFmaloifhRuFjVARbjsV-8GWETc6aNCA/viewform)


## 💬 Milestone Presentations
- [Milestone 1](https://docs.google.com/presentation/d/13reSKMykn5WhVyi5zi5oK5OygVjTZljeMWflJejQZlw/edit?slide=id.g32bc6f01e94_0_43#slide=id.g32bc6f01e94_0_43)
- [Milestone 2](https://docs.google.com/presentation/d/1suusmSruqXyRdfvViq1NKfDEqTpH5-M9w7zgh7HDCAo/edit?slide=id.g32bc6f01e94_0_74#slide=id.g32bc6f01e94_0_74)

## Setup
- Download the [NuScenes](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge) training and validation datasets, and place them together in the `data/nuscenes` directory
- Install the package requirements from requirements.txt and run ```pip install flash-attn --no-build-isolation```.
- To make sure you can download all of the tested models, authenticate your machine with huggingface using the huggingface-cli, by running ```huggingface-cli login```

## Docker

Make sure you have the correct CUDA and driver version (>=12.8) available on your system.
After building the image, run the container with the output directory mounted:

```shell
docker run --gpus all -v ./data/output:/app/data/output <img-name>
```
## Eval results

| Folder | File | accuracy | chatgpt | language/Bleu_1 | language/Bleu_2 | language/Bleu_3 | language/Bleu_4 | language/ROUGE_L | language/CIDEr | match | final_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen_Qwen2.5-VL-3B-Instruct | baseline_eval.json | 0.0 | 67.3456904541242 | 0.23925483214648655 | 0.11544553891542789 | 0.057574440433052446 | 0.020769188970051755 | 0.19079430087529606 | 0.006791738547463531 | 32.46449704142012 | 0.3542940542224235 |
| Google_Gemma-3-4b | baseline_eval.json | 0.0 | 64.21501390176088 | 0.20031983735402523 | 0.07057776733950633 | 0.020562285652540205 | 0.0065920170980479755 | 0.15465341169836444 | 0.002624376073528029 | 35.52662721893491 | 0.3432085651226965 |
| OpenGVLab_InternVL3-2B | baseline_eval.json | 0.0 | 68.50231696014829 | 0.19311353330793474 | 0.07443754271197525 | 0.02748667571393649 | 0.009723712529328537 | 0.1677772971700306 | 0.005031769222350847 | 22.62869822485207 | 0.3355647203008346 |

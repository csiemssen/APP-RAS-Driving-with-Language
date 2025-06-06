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

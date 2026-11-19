# Dog breed image classifier: comparing three pre-trained CNNs

Project for the **Udacity Future AWS AI Programmer Nanodegree** (passed August 2026).

The program uses three ImageNet-trained CNNs (VGG16, AlexNet, ResNet18) to classify 40 pet images, checks whether each image is a dog and which breed, and compares the models on accuracy and run time. Udacity provided the project template, the `classifier.py` wrapper and the image sets; I wrote the functions that build the labels, run the classifier, match the labels, compute the statistics and print the results.

## What the code does
- `get_input_args.py`: command-line arguments `--dir`, `--arch`, `--dogfile`
- `get_pet_labels.py`: builds the true labels from the image file names
- `classify_images.py`: runs the chosen CNN and compares its label with the true label
- `adjust_results4_isadog.py`: marks each label as dog or not dog using `dognames.txt`
- `calculates_results_stats.py`: counts and percentages
- `print_results.py`: prints the summary and the misclassified images
- `check_images.py`: main script, with run timing

## Results on the 40 pet images

| Model | Not-dog correct | Dog correct | Breed correct | Label match | Run time |
|---|---|---|---|---|---|
| VGG16 | 100.0% | 100.0% | 93.3% | 87.5% | 20 s |
| ResNet18 | 90.0% | 100.0% | 90.0% | 82.5% | 5 s |
| AlexNet | 100.0% | 100.0% | 80.0% | 75.0% | 2 s |

VGG16 was the most accurate on breed; AlexNet was the fastest. Full outputs are in `results/`.

## How to run
```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
sh run_models_batch.sh
```
Requires Python 3, PyTorch, torchvision and Pillow.

I used an AI assistant (Claude) as a learning aid while working on this project.

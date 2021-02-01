# Object Structure Detector
![illustration](illustration.png)
 
Structure Detector is a pose estimation network which can detect an arbitrary number of keypoints. It is based on center point detectors such as [CenterNet](https://github.com/xingyizhou/CenterNet).

# Installation
Requirements:
- Python > 3.8
- Nvidia GPU with CUDA/CUDNN installed (recommended)

We recommend using a virtual environment:
```zsh
cd StructureDetector
python3 -m venv env
source env/bin/activate
```

Install required Python packages:
```zsh
pip install -U pip
pip install -r requirements.txt
```

# Reproduce our Results
Put the validation dataset in `database/valid/` at the root directory of the repo and download the trained network.

Execute this command:

```zsh
python sources/evaluate.py --valid_dir database/valid/ --load_model model_best_classif.pth --conf_threshold 0.529 --decoder_dist_thresh 0.108
```

# Train your own Model
## Annotation
Each image should have its corresponding annotation in the same folder, with the same name. The annotation is store in JSON format with the following sructure:

```json
{
  "image_path": "database/valid/im_000000.jpg",
  "img_size": [
    2448,
    2048
  ],
  "objects": [
    {
      "label": "label",
      "parts": [
        {
          "kind": "anchor",
          "location": {
            "x": 685,
            "y": 820
          }
        },
        {
          "kind": "part_a",
          "location": {
            "x": 520,
            "y": 700
          }
        }
      ]
    }
  ]
}
```

Each annotation consists of a list of objects. Each of them has a label and a list of keypoints. An object should have exactly one keypoint of kind "anchor" and an arbitrary number of other keypoints. You can customize the anchor name in the command line arguments (`--anchor_name` option). All coordinates are in pixels relative to the top-left image corner.

You should also update the `label.json` file with the names of your labels and part kinds. Example with our dataset:

```json
{
    "labels": ["bean", "maize"],
    "parts": ["leaf"]
}
```

For annotating crops you can use [this repo](https://github.com/laclouis5/StructureAnnotator) or adapt it to your needs.

## Training
Split your dataset into two folders: one for training and the other for validation. Optionnaly launch TensorBoard to monitor training (use a secondary shell):
```zsh
tensorboard --logdir runs
```

Launch training with:
```zsh
python sources/train.py --train_dir train_dir/ --valid_dir valid_dir/
```

Customize training (epochs, learning rate, ...) by specifying options in the command line arguments. help is available:
```zsh
python sources/train.py -h
```

Best networks are saved in a `trainings/` directory created at the root directory.

# Contact
Feel free to ask you questions or request data at louis.lac@ims-bordeaux.fr.

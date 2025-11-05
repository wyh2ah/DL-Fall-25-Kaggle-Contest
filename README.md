## `main.py` is our final version for highest accuracy.

## Reproduce and model weights:
You can reproduce by `python demo.py --checkpoint_path /path/to/checkpoint/folder`. It should have an accuracy of 81.80% over the evaluation dataset. 

You can find our trained model's weights at [here](https://drive.google.com/file/d/12aTHHJtSJIm3tuEZMzqVdaKfJyCNJ_gd/view?usp=drive_link).

## `prompt.ipynb` is for tuning the best prompt.

## `finetune.py`：
We attempted to fine-tune the model using the unlabeled test split of the test dataset. However, the results were unsuccessful, and this experiment is not related to our final model.

## `wrong_case.yaml` is the collection of the wrong predictions.



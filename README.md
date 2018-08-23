# Chest-X-Ray
Here we ensemble 3 diffrent models (MobileNet, Inception, Xception) and then classified and localized x-ray images into 5 category, including:
- Normal 
- Infiltration 
- Atelectasis 
- Effusion
- Pneumothorax

# End result:

<img src="./assets/result.png" aligh="right">

# Evalutation
each class has been evaluated seperatly and the exact information can be found in notebook. Now let's take a look at the evaluation of Pneumothorax.

<img src="./assets/pneumothorax_evaluation.png>

# Usage
Run "prediction_and_localization.py" and enter the location of your xray image for getting the predictions.

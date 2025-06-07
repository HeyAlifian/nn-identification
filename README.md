# nn-classification-project
Text classification using trained neural network or machine learning python. **Still WIP, expect any bugs or errors occured, thank you!**
The model takes up to 2-5 minutes to train according to the given configurations, inputs, or sample data. You can check on the `'sample_data.json'` for a more detailed sample data that the model uses.

- **Available Labels or Categories:**
    - Greeting
    - Farewell
    - Question
    - Compliment
    - Apology
    - Disagreement
    - Agreement
    - Request
    - Emotional_Expression

### Required Packages
To install the necessary packages, try:
```bash
pip install -r requirements.txt
```
It will automatically installs all the necessary packages inside the `'requirements.txt'` including the numpy and scikit-learn.

## Note
If you ever wanted to **add even more sample data** inside the json file, make sure you deleted the current trained model `'.pkl'` file. Because when the current trained model file is not deleted, it raises an error. Meaning you have to delete the trained model file.

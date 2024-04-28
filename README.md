## Installation and Dependancies

To set up this project, you'll need Python 3.6 or higher and the following Python libraries:

- pandas
- numpy
- transformers
- scipy
- tqdm
- scikit-learn

To install the dependencies, use the following command:

`bash
pip install pandas numpy transformers scipy tqdm scikit-learn


## Usage
To use this project, run the main.py script. It will prompt you to enter a review or text input and will return the predicted star rating based on the sentiment analysis results.
Example:

`bash:

python main.py

`text:

Give a review you'd like to be predicted:
The product quality is excellent but the delivery was delayed.

The predicted score is: 4



## Model Information
The sentiment model takes from Rob Mulla, using a pre-trained RoBERTa model from Hugging Face for sentiment analysis. 
Using transfer learning from twitter -> amazon reviews, this model outputs scores for negative, neutral, and positive sentiment. 

The classifier uses the Random Forest model trained to predict star ratings based on sentiment scores.
Provided within is both the original model and the downsampled model.
Fine tuning has been done to some degree, however this model still only scores at around a 70% accuracy.
This level of accuracy is generally sufficient for applications where the distinction between star ratings (e.g., between 4-star and 5-star reviews) may not be critical. 
Further tuning or additional techniques might help improve accuracy in the future.



# Licensing 
This project is licensed under the MIT License.


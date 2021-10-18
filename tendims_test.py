import time
import numpy as np
import pandas as pd

import tendims

# load the model
model = tendims.TenDimensionsClassifier(is_cuda=True, models_dir = './models/lstm_trained_models', embeddings_dir='./embeddings')
# print the list of dimensions
dimensions = model.dimensions_list
print(dimensions)

#### test the classifier on a single sentence

# one example sentence
sentences = ["Only a fully trained Jedi Knight, with The Force as his ally, will conquer Vader and his Emperor. If you end your training now, if you choose the quick and easy path, as Vader did, you will become an agent of evil"]

# compute overall score and per-sentence scores
for s in sentences:
	dim = 'knowledge'
	score = model.compute_score(s, dim)
	score_split = model.compute_score_split(s, dim)
	print (f'{s} -- {dim}={score:.2f}')
	print (f'{s} -- {dim}={score_split}')


#### test the classifier on a larger dataset

# load a pandas dataframe
df = pd.read_csv(f'example.csv', sep='\t', encoding='utf-8')

# apply the classifier to the text column of the dataframe
for dim in dimensions:
    print(dim)
    start_time = time.time()
    f = lambda x : pd.Series(model.compute_score_split(x, dim))
    df[[f'{dim}_mean' , f'{dim}_max', f'{dim}_min', f'{dim}_std']] = df['text'].apply(f)
    end_time = time.time()
    print(f'total time = {end_time-start_time} ({(end_time-start_time)/len(df)} per entry)')

# binarization by quantile thresholding on the maximum value
quantile = 0.85
for dim in dimensions:
    quantile_thresh = np.quantile(df[f'{dim}_max'].dropna().values, quantile)
    df[f'{dim}_binary_quantile_{quantile}'] = df[f'{dim}_max'].apply(lambda x : 0 if x < quantile_thresh else 1)

# binarization by fixed thresholding on the maximum value
threshold = 0.80
for dim in dimensions:
    df[f'{dim}_binary_threshold_{threshold}'] = df[f'{dim}_max'].apply(lambda x : 0 if x < threshold else 1)

# save results to file
df.to_csv('example_dimensions.csv', sep=',', index=False)
# tendimensions
<h3>Dimensions of social exchange from social psychology</h3>

Decades of social science research identified ten fundamental dimensions that provide the conceptual building blocks to describe the nature of human relationships. The dimensions are the following:

- <b>Knowledge</b>: Exchange of ideas or information
- <b>Power</b>: Having power over the behavior and outcomes of another
- <b>Status</b>: Conferring status, appreciation, gratitude, or admiration upon another
- <b>Trust</b>: Will of relying on the actions or judgments of another
- <b>Support</b>: Giving emotional or practical aid and companionship
- <b>Romance</b>: Intimacy among people with a sentimental or sexual relationship
- <b>Similarity</b>: Shared interests, motivations or outlooks
- <b>Identity</b>: Shared sense of belonging to the same community or group community
- <b>Fun</b>: Experiencing leisure, laughter, and joy
- <b>Conflict</b>: Contrast or diverging views

These dimensions are commonly expressed in language. For example "we all congratulate her on her wonderful accomplishment during this last year" is an expression of status giving, and "if you need any help call me" is an expression of support. After annotating conversational text with these ten labels through crowdsourcing, we trained an LSTM tool to detect the presence of these types of interactions from conversations (or any piece of text, really).

<h3>References</h3>

If you use our code in your project, please cite our research papers:
- <b>[CITATION]</b> M. Choi, L.M. Aiello, K.Z. Varga, D. Quercia "Ten Social Dimensions of Conversations and Relationships". The Web Conference, 2020
- <b>[CITATION]</b> S. Deri, J. Rappaz, L.M. Aiello, D. Quercia "Coloring in the Links: Capturing Social Ties As They Are Perceived". ACM CSCW, 2018

<h3>Setup</h3>

<h4>1. Clone the repository</h4>

<code>git clone https://github.com/lajello/tendimensions.git</code>

<h4>2. Install the dependencies</h4>

<code>pip install -r requirements.txt</code>

<h4>3. Download embeddings</h4>

The classifiers use three different embeddings: word2vec, glove, and fasttext. The default location for the embedding files is the directory <code>embeddings</code>. You can change the location, but you will need to specify the new directory when instantiating the main TenDimensionsClassifier class (see examples below)

1) Word2Vec: the file <code>GoogleNews-vectors-negative300.wv</code> should be placed in the directory <code>embeddings/word2vec</code>. Download it from: https://code.google.com/archive/p/word2vec/
2) Fasttext: the file <code>wiki-news-300d-1M-subword.wv</code> should be placed in the directory <code>embeddings/fasttext</code>. Download it from: https://fasttext.cc/docs/en/english-vectors.html
3) GloVe: the file <code>wiki-news-300d-1M-subword.wv</code> should be placed in the directory <code>embeddings/glove</code>. Download it from: https://nlp.stanford.edu/projects/glove/

For your convenience, you can dowload all the embeddings at once from http://www.lajello.com/files/tendimensions_embeddings.zip (4GB+). <b>Please refer to the original pages for documentation and acknowledgments.</b>

<h3>Usage</h3>

Once the code is set up, usage is very simple.

```
import tendims
model = tendims.TenDimensionsClassifier(is_cuda=True, embeddings_dir = './embeddings') #set to False is GPU not available. Customize embeddings directory if needed
dimensions = model.dimensions_list #get the list of all available dimensions
print(dimensions)
text = 'Hello, my name is Mike. I am willing to help you, whatever you need.'
dimension = 'support'
model.compute_score_split(text, dimension) #compute a score for each sentence and returns maximum and average values
model.compute_score(text, dimension) #compute a single score on the whole text
```

A jupyter notebook with examples is included in this repository.

<h3>Tips and tricks</h3>

After following the steps above you can already play around with the tool. However, it is highly recommended to read the following guidelines thorougly to interpret the results properly.

<h4>Sentence-level classification</h4>

The classifier was trained on individual sentences. Although the classifier accepts text of any length, we recommend to compute the scores sentence-by-sentence. The function <code>compute_score_split</code> does that for you and returns the maximum and average values. When using the maximum, please consider that the longer the text, the higher the likelihood to get a larger maximum value. So, if you use the maximum, be sure to account for text length in you analysis (i.e. a high maximum score on a text of 10 words is not comparable with the same value on a text of 100 words). You can always split the sentences yourself and aggregate sentence-level values as you deem appropriate.

<h4>Score distribution</h4>

The classifier returns confidence scores in the range [0,1]. This number is proportional to the likelihood of the text containing the selected dimension. Depending on the input data and on the aggregation performed, the empirical distributions of the confidence score may differ across dimensions (may be bell-shaped, skewed, bi-modal, etc.). For this reason, binarizing the scores based on a fixed threshold might not be the best approach. An approch that proved effective is to binarize based on a high percentile (e.g., 75th or 85th percentiles) computed on your empirical distribution of scores.

<h4>Directionality</h4>

The classifier was trained to identify expressions that "convey" dimension D from the speaker to the listener. For example, in the case of the dimension <i>support</i>, the classifier is supposed to find expressions indicating that the speaker is offering some support to the lister. In practice, this directionality is not guaranteed, and the classifier picks up different types of verbal expressions of the social dimensions. For example, "I am willing to help you, whatever you need" and "Clara is willing to help George, whatever he needs" have both relatively high scores for the dimension <i>support</i> (0.86 and 0.75, respectively), but only the first one is an expression of the speaker offering support. To more strongly enforce directionality, and approach that proved effective is to consider only sentences containing second-person pronouns. 

<h4>Errors</h4>

Be aware that the classifier was trained mostly on Reddit data. It can be used on any piece of text but you should expect some performance drop when used on textual data with very different style or distribution of words (e.g., Twitter). Last, as everything in life, the classifications made by this tool are not perfect, but given eough data you'll be able to see interesting and meaningful trends. 


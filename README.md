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

These dimensions are commonly expressed in language. For example "we all congratulate her on her wonderful accomplishment during this last year" is an expression of status giving, and "if you need any help, you know, back up, call me" is an expression of support. After annotating conversational text with these ten labels through crowdsourcing, we trained an LSTM tool to detect the presence of these types of interactions from conversations (or any piece of text, really).

<h3>Setup</h3>

1) Clone the repository
2) 


<h3>Usage</h3>

Once the code is set up, usage is very simple.

<code>import tendims</code>

<code>model = tendims.TenDimensionsClassifier(is_cuda=True) #set to False is GPU not available</code>

<code>dimensions = model.dimensions_list</code>

<code>print(dimensions)</code>

<code>text = 'I love you so much, I am in love with you. Fuck you very much.'</code>

<code>dimension = ''</code>

<code>model.compute_score_split(text, dimension) #compute a score on each sentence and returns maximum and average values</code>

<code>model.compute_score(text, dimension) #compute a single score on the whole text</code>

A jupyter notebook with examples is included in this repository.

<h3>Tips and tricks</h3>

1) Usa il classifier su ogni frase di un Reddit comment, non sul commento totale. C'e' gia' una funzione che calcola facendo lo split. Se vuoi assegnare uno score a livello del commento, puoi usare average o maximum. Io di solito uso maximum perche' se anche solo una frase del messaggio contiene la dimensione, allora si puo' dire che anche il messaggio la contiene. Se vai per maximum pero' devi poi stare attento a comparare messaggi con dimensioni diverse, perche' piu' lungo il messaggio e' piu' e' probabile che ci sia almeno una frase con un valore alto e quindi se non consideri la lunghezza ti ritrovi in situazioni di Simpson's paradox :)

2) In generale, il classifier trova espressioni che esprimono una dimensione, ma non necessariamente che trasmettono quella dimensione dal sender al recipient. Per esempio: "I am willing to help you, whatever you need" and "George is willing to help Clara, whatever she needs" saranno probabilmente entrambe due frasi di "support", ma ovviamente la prima e' un'espressione di support che va dal sender al recipient e la seconda e' una descrizione di una situazione di support. Se vuoi catturare di piu' il primo caso rispetto al secondo, devi fare un filtro considerando frasi con i second person pronouns.

3) Le distribuzioni dei confidence scores del classificatore sono molto diverse tra le varie dimensioni. Questo succede per molte ragioni, non ultima il fatto che diverse dimensioni usano diversi embeddings (almeno in questa implementazione che ti ho dato). Quindi quello che ti consiglierei di fare, se il tuo experimental setup lo permette, e' di usare threshold dinamiche sul confidence score. Per esempio, per capire se un messaggio e' di support, fai prima la distribuzione degli score di support su tutto il dataset, poi prendi l'xth percentile (ho visto empiricamente che 75th, 85th funzionano meglio di altri) e decidi che il messaggio e' support se il suo confidence score supera quel percentile. Ovviamente questo non e' ideale se vuoi contare il numero di messaggi di support vs. conflict, per esempio, ma per una serie di studi comparativi (e.g., quanto support c'e' nel'user group A vs. l'user group B) questo e' un setup accettabile.

4) Ovviamente, il classifier non e' perfetto e alcune dimensioni funzionano meglio di altre, ma vedrai che se hai abbastanza dati come nel tuo caso ti verranno fuori cose meaningful :)

<h3>References</h3>

If you use our code in your project, please cite our research papers:
- <b>[CITATION]</b> M. Choi, L.M. Aiello, K.Z. Varga, D. Quercia "Ten Social Dimensions of Conversations and Relationships". The Web Conference, 2020
- <b>[CITATION]</b> S. Deri, J. Rappaz, L. M. Aiello, D. Quercia "Coloring in the Links: Capturing Social Ties As They Are Perceived". ACM CSCW, 2018

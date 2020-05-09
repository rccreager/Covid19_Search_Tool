# Covid19_Search_Tool

## Getting started

### Via Docker

The easiest way to run this package is with Docker.
1. Install [Docker](https://docs.docker.com/install/)
2. Pull the main Docker image from Docker Hub:

        docker pull rccreager/covid19-search-tool:latest-main 
3. Run the main Docker image:

        docker run -it -p 8888:8888 rccreager/covid19-search-tool:latest-main

You should now be "inside" the main Docker container at the command line and can now run a Latent Dirichlet Allocation model with TF and TF-IDF embeddings:

        python3 lda_tf_tfidf.py 

Note that you only need the `-p 8888:8888` flag in the docker run command if you plan to use Jupyter from within this container later.

**TO-DO:**
- Check sensitivity of topics to initial random seeding. 
- Check if some abstracts are in Spanish -- we sometimes get clusters of Spanish articles ('la', 'los'); how can we filter out non-English text? 

#### Optional: Use BERT embeddings

To get the BERT embeddings, you'll need to start up a second Docker container on the same machine and network it with the main container.
1. Pull the server Docker image from Docker Hub:

        docker pull rccreager/covid19-search-tool:latest-bert-server
2. Run the server image: 

        docker run -it -runtime nvidia rccreager/covid19-search-tool:latest-bert-server 1 40 

Give it a little time to build the graph and start the server. You know it's working when you see a line like: "I:WORKER-0:[\_\_i:gen:559]:ready and listening!".

Explanation of the command line flags:

The `-it` flag makes this container interactive and uses TTY to make a pseudo-terminal for you.
The `-runtime nvidia` flag enables GPU usage for this container. 
The `1` is a command line option (for the number of BERT server workers) to the `bert_service/entrypoint.sh` script running in this container.
The `40` is another command line option for `bert_service/entrypoint.sh` (for the maximum BERT sequence length).

3. To allow the BERT client in the main docker container to access the BERT server container, we must add them to the same Docker network and set the server IP address:

First, create the network:
        docker network create my-net --gateway=1.2.3.4 --subnet=1.2.3.4/11
        
Next, find the network ID of the server and client container by listing all Docker containers running on the machine:
        
        docker container ls 

Once you have those IDs, add both containers to the network, substituting in your container IDs. You only need to set an IP address for the server:

        docker network connect --ip 1.2.4.8 my-net ID-NUMBER-OF-SERVER
        docker network connect my-net ID-NUMBER-OF-CLIENT

4. Finally, you can either run the embeddings and save them to a CSV for later use: 

        python3 create_bert_embeddings.py 

Or, you can run LDA by running these embeddings:

        python3 lda_bert.py 

**TO-DO:** 
- Check if embedding CSV already found and read in. 
- Try HuggingFace BERT tools for faster embedding (is BERT-service actually using the GPU properly?). 
- Try more appropraite existing embeddings or retrain the embeddings using a more appropriate corpus.  
- Create word cloud visualizations during embedding. User BERT embeddings in BM25 search tool. 

#### Optional: Start Jupyter from Main Container

This assumes you've pulled and are running the main Docker container from the instructions at the top.
1. Start Jupyter from inside the main Docker container:

        jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
2. Open Jupyter on your local machine by copy-pasting the printed address into a web brower. It will look something like:

        http://127.0.0.1:8888/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

You can now navigate to the notebooks folder to run the BM25 notebook or add more notebooks yourself!
### Building Yourself:

Please view the Dockerfiles in the main repo and within the `bert_service` repo to see precisely how to build yourself.

To get the dataset itself (stored here via git LFS):
- Visit [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and download the data (requires Kaggle account)
- Clone this [repository](https://github.com/rccreager/Covid19_Search_Tool), move the data to Covid19_Search_Tool/data, and unzip the files



## Backup resources
[Active colab notebook](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr) : Resources for working with CORD19 (Novel Coronovirus 2019) NLP dataset -

## Interactive visualization of COVID-19 related academic articles
![Alt text](img/CORD19_Bert_Embeddings_6000_articles_in_top_journals.png?raw=true "CORD19_Bert_Embeddings_6000_articles_in_top_journals.png")
**TSNE Visualization of COVID-19 related academic articles**
- Color encodes journal
- BERT sentance embeddings are article abstracts
- Using standard BERT pre-trained model (no retraining yet)
- 6200 total articles

### Custom CORD19 NLP Search engine
![Alt text](img/CORD19_nlp_search_engine.png?raw=true "CORD19_nlp_search_engine")
- BM25 natural language search engine
- Data Processing
    1. Remove duplicate articles
    2. Remove (or annotate) non-academic articles (TODO)
- NLP Preprocessing
    1. Remove punctuations and special characters
    2. Convert to lowercase
    3. Tokenize into individual tokens (words mostly)
    4. Remove stopwords like (and, to))
    5. Lemmatize
- [Thanks DwightGunning for the great starting point here!](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr)

### Plan of action
- Topic modeling with LDA @Rachael Creager 
- NLU feature engineering with TF-IDF @Maryana Alegro 
- NLU feature engineering with BERT @Matt rubashkin
- Feature engineering with metadata
- Making an embedding search space via concatenating the TOPIC, NLU and metadata vectors @Kevin Li
- Then Creating a cosine sim search engine that creates the same datatype as the above vector
- Streamlit app that has search bar, and a way to visualize article information (Mike Lo)

### Current work based on:
- [BM 25 Search Engine by DwightGunning](https://colab.research.google.com/drive/1aFxUJgP1GeMqqw3bUDQIzoYIaYHWKCAr)
- [Building a search engine with BERT and Tensorflow](https://colab.research.google.com/drive/1ra7zPFnB2nWtoAc0U5bLp0rWuPWb6vu4)

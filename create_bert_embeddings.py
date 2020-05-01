import os, sys
from bert_serving.client import BertClient

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
src_dir = os.path.join(nb_dir,'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import local libraries
from nlp import get_preprocessed_abstract_text 

def create_bert_embeddings(data_path, data_file_name, output_name, server_ip, is_tokenized) 
    print('getting text')
    preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv')
    print(preprocessed[0])
    print('getting client')
    with BertClient(ip = '1.2.4.8', timeout='10000') as bc:
        print('encoding')
        bc.encode(preprocessed, is_tokenized=is_tokenized)
        print('fetching')
        bert_vectors = fetch_all(sort=True, concat=True)
        with open(output_name, 'w') as output_file:
            output_file.writelines("%s\n" % vector for vector in bert_vectors)
    return output_name

if __name__ == '__main__':
    create_bert_embeddings('data/CORD-19-research-challenge/', 'metadata.csv', 'bert_embeddings.txt', '1.2.4.8', False)


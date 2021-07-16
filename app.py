from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
from preprocess import prepro
# from tqdm import tqdm
from gensim.models.fasttext import FastText
# from rank_bm25 import BM25Okapi
import nmslib
import time
import numpy as np
import pickle
import re
import pandas as pd
import csv
# from predict import preprocess


app = Flask(__name__)
#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'

ft_model = FastText.load('_fasttext.model')
weighted_doc_vects = []
weighted_doc_vects = pickle.load(open("weighted_doc_vects.p", "rb" ))

df = pd.read_csv('merged.csv', error_bad_lines=False)

# create a matrix from our document vectors
data = np.vstack(weighted_doc_vects)

# initialize a new index, using a HNSW index on Cosine Similarity
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)


@app.route('/api/companies',methods = ['POST'])
@cross_origin
def get_relevant_results():

  if request.method == 'POST':
    data = request.get_json() 

    if 'title' not in data.keys():
      return jsonify(
        message='title missing',
        success='fail'
      )

    input = data['title']
    # print(input)

    # input = "ntfs-3g - Unsanitized modprobe Environment Privilege Escalation 2017"
    input = prepro(input)

    query = [ft_model[vec] for vec in input]
    query = np.mean(query,axis=0)

    t0 = time.time()
    ids, distances = index.knnQuery(query, k=10)
    t1 = time.time()
    # print(f'Searched {df.shape[0]} records in {round(t1-t0,4)} seconds \n')

    result_data = []
    for i,j in zip(ids,distances):
      # print(round(j,2))
      print(df.Title.values[i])
      result_data.append({
          'title': df.Title.values[i]
      }) 

    return jsonify(
          message='extracted successfully',
          success='ok',
          result_data = result_data
      )

if __name__ == '__main__':
    app.run()

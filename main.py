"""
Created on Fri Jun 29 08:51:21 2018

@author: hungn
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, jsonify
from flask import abort
from flask import request
from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import datetime
import scipy.sparse as sp
from sklearn.metrics.pairwise import linear_kernel
from mongoengine import *
import pickle
import os
from bson.binary import Binary
import base64
# nltk.download('stopwords')

def create_tfidf_matrix(file):
    tf_vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 3),max_df=0.6, min_df=0, stop_words='english')
    tfidf_matrix = tf_vect.fit_transform(file["Description"])
    return tf_vect, tfidf_matrix


def find_similar_in_same_matrix(tfidf_matrix, index, top_n = 3):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]


def find_similar(file, tfidf_matrix, input_string,tf_vectorizer, top_n = 3):
    string_vector = tf_vectorizer.transform(np.array([input_string]))
    cosine_similarities = linear_kernel(string_vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    top_results = [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    for (index, score) in top_results:
        i = 0
        result = file.iloc[index]
        print("Score: " + str(score))
        for column in result:
            if file.columns.values[i] != "_id":
                print(file.columns.values[i] + ": \t" + column)
            i += 1
        print("\n")

def get_recommendation(input_string, configuration):
    """
       CONFIGURATION: an array of tuples, each tuple contain 3 elements
       describing the corresponding tier name, output file, tfidf_matrix and tf_vectorizer

       Example: [(name, it_towers, tfidf_matrix_it_towers, tf_it_towers), (..,..,..),..]
    """

    print("Input String: " + input_string)
    for name, file, tfidf_matrix, tf_vectorizer in configuration:
        print("=======Tier " + str(name) + ": =========")
        find_similar(file=file,
                     tfidf_matrix=tfidf_matrix,
                     input_string=input_string,
                     tf_vectorizer=tf_vectorizer
                     )


def find_similar_json(filename, file, tfidf_matrix, input_string,tf_vectorizer, top_n = 3):
    """

    :param filename:
    :param file:
    :param tfidf_matrix:
    :param input_string:
    :param tf_vectorizer:
    :param top_n:
    :return: JSON Sample
    {
        "Tier" : filename,
        "Results": [
            {
                "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :
            }
        ]
    }
    """

    string_vector = tf_vectorizer.transform(np.array([input_string]))
    cosine_similarities = linear_kernel(string_vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    top_results = [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    output = {
        "Tier": filename,
        "Results": []
    }
    count = 0
    for (index, score) in top_results:

        i = 0
        result = file.iloc[index]
        print("Score: " + str(score))
        output["Results"].append({
            "Score" : score
        })
        for column in result:
            if file.columns.values[i] != "_id":
                output["Results"][count][file.columns.values[i]] = column
                print(file.columns.values[i] + ": \t" + column)
                i += 1
        print("\n")
        count += 1
    return output


def find_similar_json_from_database(collectionName, collection, labels, tfidf_matrix, input_string,tf_vectorizer, top_n = 3):
    """

    :param filename:
    :param file:
    :param tfidf_matrix:
    :param input_string:
    :param tf_vectorizer:
    :param top_n:
    :return: JSON Sample
    {
        "Tier" : filename,
        "Results": [
            {
                "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :

            },
            {
            "Score": ,
                "..." : ,
                "Description" :
            }
        ]
    }
    """

    string_vector = tf_vectorizer.transform(np.array([input_string]))
    cosine_similarities = linear_kernel(string_vector, tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1]]
    top_results = [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

    output = {
        "Tier": collectionName,
        "Results": []
    }
    count = 0
    for (index, score) in top_results:

        i = 0
        # Use Cursor to get to that result location in database
        result = collection.find_one({"_id" : str(index)})
        output["Results"].append({
            "Score" : score
        })
        for label in labels:
            output["Results"][count][label] = result[label]
            i+= 1
        # for column in result:
        #     if file.columns.values[i] != "_id":
        #         output["Results"][count][file.columns.values[i]] = column
        #         print(file.columns.values[i] + ": \t" + column)
        #         i += 1
        # print("\n")
        count += 1
    return output


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime.datetime):
            return str(o.strftime("%Y-%m-%d %I:%M:%S %p"))
        return json.JSONEncoder.default(self, o)

"""
#===================== READ EXCEL FILE ========================================
it_towers = pd.read_excel("IT_Towers.xlsx")
services = pd.read_excel("Services.xlsx")
cost_pools = pd.read_excel("CostPools.xlsx")
"""

#===================== READ FROM DATABASE ====================================

# from bson.objectid import ObjectId
# import pprint
client = MongoClient('localhost', 27017)

db = client['rigil']
#
# services_db = db['Services']
# costPools_db = db["CostPools"]
# itTowers_db = db["IT_Towers"]
# #Convert MongoDB collection to Dataframe to work
# services = pd.DataFrame(list(services_db.find()))
# cost_pools = pd.DataFrame(list(costPools_db.find()))
# it_towers = pd.DataFrame(list(itTowers_db.find()))



#Test data
""" No test data yet. But should able to use the real interaction of user
    to create test data and verify the accuracy of the model"""
#
# #***********************************IT TOWERS MODEL***************************
# #Build the model from sklearn
# tf_it_towers, tfidf_matrix_it_towers = create_tfidf_matrix(it_towers)
# #***********************************SERVICES MODEL***************************
# tf_services, tfidf_matrix_services = create_tfidf_matrix(services)
# #***********************************COST POOLS MODEL***************************
# tf_cost_pools, tfidf_matrix_cost_pools = create_tfidf_matrix(cost_pools)
#
# #Find similar
# find_similar(file=it_towers,
#              tfidf_matrix=tfidf_matrix_it_towers,
#              input_string= "My database has troubled connecting to the server",
#              tf_vectorizer=tf_it_towers)
#
#
# #result
"""
    IO TEST FOR METHOD


"""
"""while True:
    input_string = input("Enter your description to classify: ")
    get_recommendation(input_string=input_string,
                   configuration=[
                           ("IT Towers",
                            it_towers,
                            tfidf_matrix_it_towers,
                            tf_it_towers,
                                   ),
                            ("Services",
                            services,
                            tfidf_matrix_services,
                            tf_services,
                                   ),
                            ("Cost Pools",
                            cost_pools,
                            tfidf_matrix_cost_pools,
                            tf_cost_pools,
                                   )
                           ])

      """


# jsonoutput = find_similar_json("IT Towers", it_towers, tfidf_matrix_it_towers, "Company needs financial aid for database infrastructure", tf_it_towers, top_n = 3)
# print(jsonoutput)

#======================FLASK API SERVICE ====================================


app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': "Use Post method to get recommendation ",
        'description': "JSON input with string to recommend with description property",
    }
]


@app.route('/', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})




#
# @app.route('/home', methods=['POST'])
# def create_task():
#     if not request.json or not 'description' in request.json:
#         abort(400)
#     input_string = request.json['description']
#     results = []
#     results.append(find_similar_json("IT Towers",
#                                      it_towers,
#                                      tfidf_matrix_it_towers,
#                                      input_string,
#                                      tf_it_towers,
#                                      top_n=3))
#     results.append(find_similar_json("Services",
#                                      services,
#                                      tfidf_matrix_services,
#                                      input_string,
#                                      tf_services,
#                                      top_n=3))
#     results.append(find_similar_json("Cost Pools",
#                                      cost_pools,
#                                      tfidf_matrix_cost_pools,
#                                      input_string,
#                                      tf_cost_pools,
#                                      top_n=3))
#
#     return jsonify({'data': results}), 201
#

# ===================== Document Class to be saved to MongoDB ============

@app.route('/data', methods=['POST'])
def post_data():
    """
    PARAMS: Should have the JSON body object in the form of:
    {
        name:   STRING,
        description: [], <= column names?
        label: [], <= the label that the user want to recommend later on
        threshold: INT, <= used to mark the threshold to start switching to another recommendation algorithm
        CollectionName: STRING, <= name to be store in database
        data: [] <=== should be an array of JSON objects describing the data to be saved to the database
    }


    :return:
        - Created 4 corresponding databases to that taxonomy ( i.e: taxonomy, taxonomy_corpus, taxonomy_meta,
            taxonomy_tfidf)
        - Create a vectorizer to transform the input into a vector to get recommendation
        - Return a JSON indicating status and the names of the databases that created
    """
    if not request.json:
        abort(400)

    req = request.json
    name = req['name']
    description = req['description']
    label = req['label']
    threshold = req['threshold']
    collectionName = req['collectionName']
    data = req['data']
    # ============= check if that collection already exists ========================
    if collectionName in db.collection_names():
        return jsonify({
            "Status" : "400",
            "Description": "Collection already exists in database. Use option append if wanted to append data in "
                           "collection"

        })


    # ====   Save Data to Taxonomy Database  ===
    taxonomy_collection = db[collectionName]

    # Before saving, loop through the input columns and concatenate into 1 data column to be an input to tfidf
    data_to_vectorize = []
    _id = 0 #Assign ID to each document in the collection instead of the default ID from mongoDB
    for jsonobject in data:
        jsonobject["_id"] = str(_id)
        _id += 1
        jsonobject['input_data'] = ""
        for input_description in description:
            jsonobject['input_data'] += input_description + " " + jsonobject[input_description]
        data_to_vectorize.append(jsonobject['input_data'])

    # Save to DB
    taxonomy_collection.insert_many(data)

    #    Save Metadata to Database
    taxonomy_meta = db[collectionName + "_meta"]
    taxonomy_meta.insert_one({
        "Name": name,
        "Description": description,
        "Label": label,
        "Threshold": threshold,
        "Collection Name": collectionName,
        "Time": datetime.datetime.utcnow()
    })
    #     Create Corpus
    taxonomy_corpus = db[collectionName + "_corpus"]
    for taxonomy_entry in taxonomy_collection.find():
        taxonomy_corpus.insert_one({
            "Description" : taxonomy_entry['input_data'],
            "Label" : str(taxonomy_entry['_id'])
        })

    # Create TFIDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_df=0.6, min_df=0, stop_words='english')


    # Create TFIDF matrix and save matrix and vectorizer into Database
    tfidf_matrix = vectorizer.fit_transform(raw_documents= data_to_vectorize)

    # Save vectorizer to pickle
    pickle.dump(vectorizer, open(collectionName + "_vectorizer.pickle", "wb"))
    sp.save_npz(collectionName + "_tfidf_matrix", tfidf_matrix)
    """
    idfs = vectorizer.idf_.tolist()
    vocabulary = vectorizer.vocabulary_

    # Because the default value of index of vocabulary is in np int32 form so have to change to native int type
    # to save as JSON in database
    for key,value in vocabulary.items():
        vocabulary[key] = int(value)


    taxonomy_tfidf_db = db[collectionName + "_tfidf_data"]
    taxonomy_tfidf_db.insert_one({
        "tfidf_matrix": tfidf_matrix.toarray().tolist(),
        "idfs" : idfs,
        "vocabulary": vocabulary
    })

    """
    return jsonify({
        "Status": "Success",
        "Taxonomy Collection Name": collectionName,
        "Taxonomy Meta Name": collectionName + "_meta",
        "Taxonomy Corpus": collectionName + "_corpus"
    })

# ====================== User Recommendation Service =====================================


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    PARAMS:
    {
        taxonomy : [ {
            name: "",
            labels: [
            {
                label: "",
                value: ""

            },
            {
                label: "",
                value: ""
            }]

        }
        ],
        input : [],
        data : {} <= JSON object of the thing that we want to recommend

    }


    :return:
    The same JSON array like the data params but with added extra results props for each JSON

    """
    if not request.json:
        abort(400)
    req = request.json
    taxonomies = req['taxonomy']
    input = req['input']
    data = req['data']
    input_to_recommend = ""
    results = []
    for field in input:
        input_to_recommend += data[field]


    for taxonomy in taxonomies:
        taxonomyname = taxonomy['name']
        labels = [label['label'] for label in taxonomy['labels']]


    #     Get taxonomy corpus
        corpus = taxonomyname + "_corpus"
        meta = taxonomyname + "_meta"

        threshold = db[meta].find_one({"Collection Name" : taxonomyname})['Threshold']
        if db[corpus].count() >= threshold:
            #         Corpus Exceed Threshold
            #         ====> Use Naive Bayes from the corpus data to recommend

            taxonomy_corpus_db = db[corpus]
            corpus_data = []
            for entry in taxonomy_corpus_db.find():
                corpus_data.append(entry)
        #     Check if there is already a pickled naive bayes model exist in local storage
            if os.path.exists(taxonomyname + "_naive_bayes_model.pickle"):
                with open(taxonomyname + "_naive_bayes_model.pickle", 'rb') as file:
                    pickle_model = pickle.load(file)
                with open(taxonomyname + "_corpus_vectorizer.pickle", "rb") as f:
                    vectorizer = pickle.load(f)
                input_vectorized = vectorizer.transform(np.array([input_to_recommend]))
                label_predicted = pickle_model.predict_proba(input_vectorized)[0]
                classes = list(pickle_model.classes_)
                answer = list(zip(classes, label_predicted))
                # Get the top 3
                answer.sort(key=lambda x: x[1], reverse=True)
                answer = answer[:3]

        #       Match the top 3 to a label in taxonomy
                taxonomy_table = db[taxonomyname]
                taxonomy_json = {
                    "Results": [],
                    "Tier": taxonomyname
                }
                for label_id, score in answer:
                    # Look up for that label ID in the database to retrieve the description and labels
                    label_entry = taxonomy_table.find_one({"_id": label_id})
                    print("Found entry!")
                    if label_entry is None:
                        print("No similar prediction")
                    else:
                        entry_json = {
                            "Score": score
                        }
                        for label in labels:
                            print("Label" + str(label))
                            print("Label entry: " + str(label_entry[label]))
                            entry_json[label] = label_entry[label]
                        taxonomy_json["Results"].append(entry_json)
                results.append(taxonomy_json)
            else:
                print("No ML model found in local storage. Check or retrain new model")
                # abort(400)

        else:
    #Use normal TFIDF on the original data table of the taxonomy
            # taxonomy_tfidf_db = db[taxonomyname + "_tfidf_data"]
            # cursor = taxonomy_tfidf_db.find()
            # for item in cursor:
            #     tfidf_data = item
            #
            # tfidf_matrix = np.array(tfidf_data['tfidf_matrix'])
            # idfs = np.array(tfidf_data['idfs'])
            # vocabulary = tfidf_data['vocabulary']
    #   ===== Potential Bug: might need to convert tfidf matrix into sparse matrix
    #         Potential Bug: Might need to convert vocabulary from int to int32
    #         vectorizer = CustomVectorizer(idfs, vocabulary)
    #         vectorizer._tfidf._idf_diag = sp.spdiags(idfs,
    #                                                  diags=0,
    #                                                  m=len(idfs),
    #                                                  n=len(idfs))
            with open(taxonomyname + "_vectorizer.pickle", "rb") as f:
                vectorizer = pickle.load(f)


            tfidf_matrix = sp.load_npz(taxonomyname + "_tfidf_matrix.npz")

            results.append(find_similar_json_from_database(collectionName=taxonomyname,
                                                     collection=db[taxonomyname],
                                                     labels=labels,
                                                     tfidf_matrix=tfidf_matrix,
                                                     tf_vectorizer=vectorizer,
                                                     input_string=input_to_recommend
                                                     ))

    return jsonify({'data': results}), 201

@app.route('/automatic-naivebayes', methods=['POST'])
def recommend_naive_bayes():
    """
    PARAMS:
    {
            datatablename : ""
    }


    :return:
    The same JSON array like the data params but with added extra results props for each JSON

    """
    if not request.json:
        abort(400)
    datatablename = request.json['datatablename']
    datatable = db[datatablename + "_data"]
    datatable_meta = db[datatablename + "_meta"]
    meta_config = datatable_meta.find_one({"Table Name" : datatablename})

    # taxonomies = meta_config['Taxonomy Label']
    input_field = meta_config['Input Fields']

    masked_label_to_update = []
    for data in datatable.find():
        _id = data['_id']
        input_to_recommend = ""
        results = []
        taxonomies = data["label"]


        for taxonomy in taxonomies:
            taxonomyname = taxonomy['Taxonomy name']
            # Getting the input field to concat to make a vector to make a prediction

            # Getting the label field to recommend
            for field in input_field:
                input_to_recommend += data[field]

            labels = [label['Label name'] for label in taxonomy['labels']]

            #     Get taxonomy corpus
            corpus = taxonomyname + "_corpus"
            meta = taxonomyname + "_meta"

            taxonomy_corpus_db = db[corpus]
            corpus_data = []
            for entry in taxonomy_corpus_db.find():
                corpus_data.append(entry)
                #     Check if there is already a pickled naive bayes model exist in local storage
            if os.path.exists(taxonomyname + "_naive_bayes_model.pickle"):
                with open(taxonomyname + "_naive_bayes_model.pickle", 'rb') as file:
                    pickle_model = pickle.load(file)
                with open(taxonomyname + "_corpus_vectorizer.pickle", "rb") as f:
                    vectorizer = pickle.load(f)
                input_vectorized = vectorizer.transform(np.array([input_to_recommend]))
                label_predicted = pickle_model.predict_proba(input_vectorized)[0]
                classes = list(pickle_model.classes_)
                answer = list(zip(classes, label_predicted))
                # Get the most accurate label
                answer.sort(key=lambda x: x[1], reverse=True)
                # Using a list to iterate later, using list because later on we can have flexibility to change to top k
                answer = answer[:1]

                #       Match the top 3 to a label in taxonomy
                taxonomy_table = db[taxonomyname]
                # taxonomy_json = {
                #     "Results": [],
                #     "Tier": taxonomyname
                # }
                labels_updated = []
                for label_id, score in answer:
                    # Look up for that label ID in the database to retrieve the description and labels
                    label_entry = taxonomy_table.find_one({"_id": label_id})
                    print("Found entry!")
                    if label_entry is None:
                        print("No similar prediction")
                    else:
                        # entry_json = {
                        #     "Score": score
                        # }
                        # NOTICE: CAN OPTIMIZE THIS PART SINCE WE ARE REPEATEDLY CHECK FOR CORRECT LABEL TO POPULATE
                        for label in labels:
                            # Start populating value in matching label
                            for label_to_populate in taxonomy['labels']:
                                if label_to_populate['Label name'] == label:
                                    # Update in database
                                    print("Label to populate: " + str(label) )
                                    print("Value: " + str(label_entry[label]))
                                    # label_to_populate['Value'] = label_entry[label]
                                    labels_updated.append({
                                        "Label name" : label,
                                        "Value" : label_entry[label]
                                    })


                                        # datatable.update_one(
                                        #     {
                                        #         "_id": ObjectId(_id)
                                        #     },
                                        #     {
                                        #         "$set": {"label.$[taxonomy].labels.$[labelname].Value": label_entry[label]}
                                        #     },
                                        #     {
                                        #         "$arrayFilters": [{"taxonomy.Taxonomy name" : taxonomyname},
                                        #                           {"labelname.Label name" : label}]
                                        #     }
                                        # )
                                    # except Exception:
                                    #     print("Cannot find similar data in datatable")
                                    #     # abort(400)
                masked_label_to_update.append({
                    "Taxonomy name" : taxonomyname,
                    "labels" : labels_updated
                })
                # results.append(masked_label_to_update)
                try:
                    datatable.update_one(
                        {
                            "_id": ObjectId(_id)
                        },
                        {
                            "$set": {"label": masked_label_to_update}
                        }
                    )
                except Exception:
                    print("Cannot find similar data in datatable")
                    # abort(400)
                # results.append(taxonomy_json)
            else:
                print("No ML model found in local storage. Check or retrain new model")
                # abort(400)



    return jsonify({'data': masked_label_to_update,
        "Status" : "good to go!!!!"}), 201





# =====================================================================================================================
@app.route('/datatable', methods=['POST'])
def save():
    """
   PARAMS:
   {
        tablename: ""
       taxonomy : [ {
           name: "",
           labels: []
       }
       ],
       input : [],
       data : {} <=  array of JSON objects of the thing that we want to recommend

   }


   :return:
   Generate data_meta and data_table in the backend

   """
    if not request.json:
        abort(400)
    req = request.json
    tablename = req['tablename']
    taxonomies = req['taxonomy']
    input_fields = req['input']
    datatable = req['data']

#     Create the taxonomies array with the respective field and value to be filled in for user to categorize
    taxonomy_array = []
    for taxonomy in taxonomies:
        labels = []
        for label in taxonomy['labels']:
            labels.append({
                "Label name": label,
                "Value": ""
            #     Value is Blank originally and then user will start filling in the value of the category
            })
        taxonomy_array.append({
            "Taxonomy name": taxonomy['name'],
            "labels" : labels

        })
#     Saving Data Meta
    data_meta = {
        "Table Name": tablename,
        "Taxonomy Label": taxonomies,
        "Input Fields": input_fields

    }
    if tablename + "_meta" in db.collection_names():
        return jsonify({
            "Status" : "400",
            "Description": "Table name already exists in database. Use option append if wanted to append data in "
                           "collection or choose different tablename"

        })
    data_meta_db = db[tablename + "_meta"]
    data_meta_db.insert_one(data_meta)

#   Saving Data Table
    i = 0
    for data in datatable:
        datatable[i]['label'] = taxonomy_array
        i += 1

    datatable_db = db[tablename + "_data"]
    datatable_db.insert_many(datatable)

    return jsonify({
        "Status" : "Success",
        "Datatable Name": tablename + "_data",
        "Datatable meta" : tablename + "_meta"
    })

# Saving Data to corpus after the user choose the correct category
@app.route('/savechoice', methods=['POST'])
def savechoice():
    """
    PARAMS:

    {
        tablename : "",
        _id : ObjectId(),
        label : [ {
            Taxonomy name: "",
            labels: [
                {
                    Label name: "",
                    Value: ""

                },
                {
                    Label name: "",
                    Value: ""
                }]

        }
        ]

    }


  :return:
  Insert data in the corpus and update value in the datatable and returns taxonomy meta with boolean thresholdIsPassed
  for the front end to decide which recommend service to choose

  """
    if not request.json:
        abort(400)
    req = request.json
    tablename = req['tablename']
    _id = req['_id']
    labels_chosen = req['label']
    input_to_recommend = ""
    results = []

#     Look up data in datatable to update
    datatable_db = db[tablename + "_data"]
    datatable_meta = db[tablename + "_meta"]
    meta_config = datatable_meta.find_one({"Table Name" : tablename})
    # Input to save in corpus
    input_fields = meta_config['Input Fields']
#     Search for this entry in the datatable to update the label chosen
    entry = datatable_db.find_one({"_id" : ObjectId(_id)})
    try:

        datatable_db.update_one(
            {
                "_id": ObjectId(_id)
            },
            {
                "$set": {"label": labels_chosen}
            }
        )
    except Exception:
        print("Cannot find similar data in datatable")
        abort(400)

#     Get the input to be added to the corpus and its corresponding label to corresponding corpus
    for input_field in input_fields:
        input_to_recommend += entry[input_field] + " . "

    for taxonomy in labels_chosen:
        taxonomy_json = {"thresholdIsPassed" : False}
        taxonomy_name = taxonomy['Taxonomy name']
        taxonomy_table = db[taxonomy_name]
        taxonomy_meta_config = db[taxonomy_name + "_meta"].find_one({"Collection Name": taxonomy_name})
#         Get corpus
        corpus = db[taxonomy_name + "_corpus"]
        # Look up the corresponding ID in taxonomy table
        config_match = {}
        for matched_label in taxonomy['labels']:
            config_match[matched_label['Label name']] = matched_label['Value']
        taxonomy_json['thresholdIsPassed'] = True if  corpus.count() >= taxonomy_meta_config['Threshold'] else False
        label_ID = taxonomy_table.find_one(config_match)['_id']
        corpus.insert_one({
            "Description": input_to_recommend,
            "Label": label_ID
        })
        if corpus.count() == taxonomy_meta_config['Threshold']:
            #         Threshold is passed, start training the model for the first time
            # Create TFIDF vectorizer

            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_df=0.6, min_df=0, stop_words='english')

            # Read Description from corpus and use that to fit the vectorizer
            data = pd.DataFrame(list(corpus.find()))
            x_train = data['Description']
            y_train = data['Label']

            # Create TFIDF matrix and save matrix and vectorizer into Database
            tfidf_matrix = vectorizer.fit_transform(raw_documents=x_train)
            # Training the Naive Bayes model
            NBclassifier = MultinomialNB().fit(tfidf_matrix, y_train)
            # Save the naive bayes model
            with open(taxonomy_name + "_naive_bayes_model.pickle",'wb') as file:
                pickle.dump(NBclassifier, file)
            # Save vectorizer to pickle
            pickle.dump(vectorizer, open(taxonomy_name + "_corpus_vectorizer.pickle", "wb"))
        if corpus.count() > taxonomy_meta_config['Threshold'] and corpus.count() % 100 == 0:
            #         Retrain the whole model
            # Create TFIDF vectorizer
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_df=0.6, min_df=0, stop_words='english')

            # Read Description from corpus and use that to fit the vectorizer
            data = pd.DataFrame(list(corpus.find()))
            x_train = data['Description']
            y_train = data['Label']

            # Create TFIDF matrix and save matrix and vectorizer into Database
            tfidf_matrix = vectorizer.fit_transform(raw_documents=data)
            # Training the Naive Bayes model
            NBclassifier = MultinomialNB().fit(x_train, y_train)
            # Save the naive bayes model
            with open(taxonomy_name + "_naive_bayes_model.pickle",'wb') as file:
                pickle.dump(NBclassifier, file)
            # Save vectorizer to pickle
            pickle.dump(vectorizer, open(taxonomy_name + "_corpus_vectorizer.pickle", "wb"))
        taxonomy_json['Taxonomy Meta'] = taxonomy_meta_config
        results.append(taxonomy_json)
    return JSONEncoder().encode(results)

# ======================================= GET REQUEST =====================================================

# Get Taxonomy Data
@app.route('/taxonomy/<name>', methods=['GET'])
def displaytaxonomy(name):
    """
    PARAMS:
        name : string ===> name of taxonomy in database


  :return:
  JSON array of all entries in that taxonomy

  """
    taxonomy_db = db[name]
    results = []
    for entry in taxonomy_db.find():
        results.append(entry)
    return jsonify(results)


# Get Taxonomy Meta
@app.route('/taxonomy-meta/<name>', methods=['GET'])
def displaytaxonomymeta(name):
    """
    PARAMS:
        name : string ===> name of taxonomy in database


  :return:
  JSON array of all entries in that taxonomy meta

  """
    taxonomy_meta_db = db[name + "_meta"]
    results = []
    for entry in taxonomy_meta_db.find():
        results.append(entry)
    return JSONEncoder().encode(results)


# Get Taxonomy Corpus
@app.route('/taxonomy-corpus/<name>', methods=['GET'])
def displaytaxonomycorpus(name):
    """
    PARAMS:
        name : string ===> name of taxonomy in database


  :return:
  JSON array of all entries in that taxonomy

  """
    taxonomy_corpus_db = db[name + "_corpus"]
    results = []
    for entry in taxonomy_corpus_db.find():
        results.append(entry)
    return JSONEncoder().encode(results)


# Get Datatable Data
@app.route('/datatable/<name>', methods=['GET'])
def displaydatatable(name):
    """
    PARAMS:
        name : string ===> name of taxonomy in database


  :return:
  JSON array of all entries in that taxonomy

  """
    datatable_db = db[name + "_data"]
    results = []
    for entry in datatable_db.find():
        results.append(entry)
    return JSONEncoder().encode(results)


# Get Datatable meta
@app.route('/datatable-meta/<name>', methods=['GET'])
def displaydatatablemeta(name):
    """
    PARAMS:
        name : string ===> name of taxonomy in database


  :return:
  JSON array of all entries in that taxonomy

  """
    datatablemeta_db = db[name + "_meta"]
    results = []
    for entry in datatablemeta_db.find():
        results.append(entry)
    return JSONEncoder().encode(results)


if __name__ == '__main__':
    app.run(debug=True)


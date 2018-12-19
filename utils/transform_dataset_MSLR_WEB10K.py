import pandas as pd
from collections import defaultdict
import argparse
import os


def get_doc_details(doc_string):
    doc_ls = doc_string.split(" ")
    return {"doc_id": doc_ls[2], "inc": doc_ls[5], "prob": doc_ls[8]}


def get_document_featues(feature_ls):
    d_ls = {f.split(":")[0]: f.split(":")[1] for f in feature_ls}
    return d_ls


def get_line_features(line):
    # Z,doc_details_str = line.split("#")
    z_ls = line.split(" ")
    y, x = z_ls[0], z_ls[1:]
    doc_f = get_document_featues(x)
    json_d = {"y": y, **doc_f}
    return json_d


def parse_file(data_file):
    query_d = defaultdict(list)
    with open(data_file) as fp:
        line = fp.readline()
        cnt = 0
        while line:
            strip_line = line.strip()
            q_json = get_line_features(strip_line)
            query_d[q_json['qid']].append(q_json)
            if cnt % 10000 == 0:
                print("line cnt :{}".format(cnt))
            line = fp.readline()
            cnt += 1

    return query_d


def create_data_dir(doc_json, output_dir):
    """
        dumps data in the form output_dir/q_id.csv

    """
    os.makedirs(output_dir, exist_ok=True)
    for q_id, query in doc_json.items():
        q_df = pd.DataFrame(query)
        q_df = q_df.sort_values(by=['y'], ascending=False)
        q_df.to_csv("{}/{}.csv".format(output_dir, q_id), index=None)


def get_metafile(doc_json):
    """
        returns metafile of the doc_json
    """
    q_meta_ls = []
    for q_id, query in doc_json.items():
        q_d = {"qid": q_id, "num_docs": len(query)}
        q_meta_ls.append(q_d)

    return pd.DataFrame(q_meta_ls)


# Training settings
parser = argparse.ArgumentParser(description='Transform Dataset')
parser.add_argument('--data_file', default="../data/MSLR-WEB10K/Fold1/test.txt", type=str, help='data file')
parser.add_argument('--output_dir', default="../data/MSLR-WEB10K/Fold1/test_json", type=str, help='output_dir file')
args = parser.parse_args()

data_file = args.data_file
output_dir = args.output_dir

print("Transforming File {}".format(data_file))

doc_json = parse_file(data_file)
create_data_dir(doc_json,output_dir)
mdf = get_metafile(doc_json)
mdf.to_csv("{}/{}.csv".format(output_dir,"metafile"),index=None)
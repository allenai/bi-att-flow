import csv
import json
from collections import OrderedDict
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("tsv_path")
    return parser.parse_args()


def json2tsv(json_path, tsv_path):
    configs = json.load(open(json_path, 'r'), object_pairs_hook=OrderedDict)
    type_dict = OrderedDict([('id', 'str')])
    for id_, config in configs.items():
        for key, val in config.items():
            if val is None:
                if key not in type_dict:
                    type_dict[key] = 'none'
                continue

            type_name = type(val).__name__
            if key in type_dict and type_dict[key] != 'none':
                assert type_dict[key] == type_name, "inconsistent param type: %s" % key
            else:
                type_dict[key] = type_name

    with open(tsv_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(type_dict.keys())
        writer.writerow(type_dict.values())
        for id_, config in configs.items():
            config["id"] = id_
            row = [config[key] if key in config and config[key] is not None else "None"
                   for key in type_dict]
            writer.writerow(row)


def main():
    args = get_args()
    json2tsv(args.json_path, args.tsv_path)


if __name__ == "__main__":
    main()

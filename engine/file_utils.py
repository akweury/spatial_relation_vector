# Created by shaji on 04-Jan-23
import json
def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data
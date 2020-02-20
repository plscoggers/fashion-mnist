import json

experiment_data = []

with open('results.txt') as json_file:
    data = json.load(json_file)
    experiment_data = [{
                        'batch_size': d['batch_size'],
                        'pooling': d['pooling'], 
                        'with_bn': 1 if d['with_bn'] else 0, 
                        'with_dropout': 1 if d['with_dropout'] else 0, 
                        'acc': d['test_accs'][-4], 
                        }
                    for d in data]

with open('results_parsed.txt', 'w') as file:
    file.write(json.dumps(experiment_data))
        
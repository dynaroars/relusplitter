import os
import sys
import sqlite3

from pathlib import Path

from experiment.benchmarks import *
from helpers import *

tool_root = Path(os.environ['TOOL_ROOT'])


db = sqlite3.connect(tool_root/"experiment/dbs/init_veri.db")
cursor = db.cursor()


benchmark = benchmarks[sys.argv[1]]
benchmark_name = benchmark['name']

# ================================================
# check for unknown status
query = '''
SELECT DISTINCT status AS distinct_status_count
FROM my_table;
'''

# Execute the query
cursor.execute(query)

# Fetch the result
distinct_status_count = cursor.fetchall()

# Print the number of distinct statuses
print(f'Distinct statuses: {distinct_status_count}')

# ================================================
# check all instances x verifier has three repeats
# and all repeats give the same results
query = '''
SELECT benchmark, onnx, vnnlib, verifier
FROM my_table
GROUP BY benchmark, onnx, vnnlib, verifier
HAVING COUNT(repeat) != 3 OR COUNT(DISTINCT status) > 1;
'''
# Execute the query
cursor.execute(query)
# Fetch all results
results = cursor.fetchall()
if len(results) > 0:
    print(f"Error: {len(results)} instances x verifier do not have three repeats or do not give the same results")
    for result in results:
        print(result)
    sys.exit(1)

# ================================================
# instance x verifier with average runtime across repeats
query = '''
SELECT 
    benchmark, 
    onnx, 
    vnnlib, 
    verifier, 
    status, 
    AVG(time) AS avg_time
FROM 
    my_table
WHERE 
    (benchmark, onnx, vnnlib, verifier) IN (
        SELECT benchmark, onnx, vnnlib, verifier
        FROM my_table
        GROUP BY benchmark, onnx, vnnlib, verifier
        HAVING COUNT(repeat) = 3 AND COUNT(DISTINCT status) = 1
    )
GROUP BY 
    benchmark, 
    onnx, 
    vnnlib, 
    verifier, 
    status;
'''

# Execute the query
cursor.execute(query)

# Fetch all results
# (benchmark, onnx, vnnlib, verifier, status, time)
results = cursor.fetchall()
results  = [i for i in results if i[0] == benchmark_name]



instance_x_verifier = {}
for i in results:
    benchmark, onnx, vnnlib, verifier, status, time = i
    if (onnx, vnnlib) not in instance_x_verifier:
        instance_x_verifier[(onnx, vnnlib)] = {verifier: (status,time)}
    else:
        instance_x_verifier[(onnx, vnnlib)][verifier] = (status,time)

usable_instances = set()
if benchmark_name == "tllverifybench":
    usable_instances = set(instance_x_verifier.keys())
else:
    for i in instance_x_verifier:
        results = [ res[0] for res in instance_x_verifier[i].values()]
        if all( [ status in ['sat','unsat'] for status in results]) and len(set(results)) == 1:
            # only keep instance where all verifiers agree on the result
            usable_instances.add(i)
        else:
            print(f"{i} was removed for {[ (verifier, res[0].upper() if res not in ['sat', 'unsat'] else res[0]) for verifier,res in instance_x_verifier[i].items()]}")
print(f"Number of instances: {len(instance_x_verifier)}")
print(f"Number of usable instances: {len(usable_instances)}")
        

fname = tool_root/f"experiment/selected_instances_{benchmark_name}.csv"

with open(fname, 'w') as f:
    f.write('onnx,vnnlib,result, v1, t1, v2, t2, v3, t3\n')
    for i in usable_instances:
        onnx, vnnlib = i
        result = list(instance_x_verifier[i].values())[0][0]
        (v1,s1,t1), (v2,s2,t2), (v3,s3,t3) = [ (verifier, status, time) for verifier, (status, time) in instance_x_verifier[i].items()]
        f.write(f'{onnx},{vnnlib},{result},{v1},{s1},{t1},{v2},{s2},{t2},{v3},{s3},{t3}\n')

db.close()
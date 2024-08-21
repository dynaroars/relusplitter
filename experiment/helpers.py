import os
import subprocess
import sqlite3
from pathlib import Path
from multiprocessing import cpu_count

from relu_splitter.verify import init_verifier

def get_init_veri_db(db_path):
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS my_table (
        benchmark TEXT,
        onnx TEXT,
        vnnlib TEXT,
        verifier TEXT,
        repeat INTEGER,
        status TEXT,
        time REAL,
        PRIMARY KEY (benchmark, onnx, vnnlib, verifier, repeat)
    )
''')
    db.commit()
    return db

def already_in_veri_db(db, benchmark, onnx, vnnlib, verifier, repeat):
    cursor = db.cursor()
    query = '''
        SELECT 1 FROM my_table WHERE
        benchmark = ? AND
        onnx = ? AND
        vnnlib = ? AND
        verifier = ? AND
        repeat = ?
    '''
    cursor.execute(query, (benchmark, onnx, vnnlib, verifier, repeat))
    result = cursor.fetchone()
    return result is not None

def insert_into_veri_db(db, benchmark, onnx, vnnlib, verifier, repeat, status, time):
    if not already_in_veri_db(db, benchmark, onnx, vnnlib, verifier, repeat):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO my_table (
                benchmark, onnx, vnnlib, verifier, repeat, status, time
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (benchmark, onnx, vnnlib, verifier, repeat, status, time))
        db.commit()

def get_veri_db(db_path):
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS instances (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark TEXT,
        onnx TEXT,
        vnnlib TEXT,
        seed INTEGER,
        split_idx INTEGER,
        nsplits INTEGER,
        mask TEXT,
        strat TEXT,
        log TEXT,
        output TEXT,
        UNIQUE(benchmark, onnx, vnnlib, seed, split_idx, nsplits, mask, strat)
    )
''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS verification_result (
        verifier TEXT,
        status TEXT,
        time REAL,
        log TEXT,
        foreign key (problem) references instances(id),
        PRIMARY KEY (verifier, problem)
    )
''')
    db.commit()
    return db

def instance_already_generated(db, instance):
    cursor = db.cursor()
    query = '''
        SELECT 1 FROM instances WHERE
        benchmark = ? AND
        onnx = ? AND
        vnnlib = ? AND
        seed = ? AND
        split_idx = ? AND
        nsplits = ? AND
        mask = ? AND
        strat = ?
    '''
    cursor.execute(query, instance)
    result = cursor.fetchone()
    return result is not None


def insert_into_instance_db(db, instance):
    if not instance_already_generated(db, instance):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO instances (
                benchmark, onnx, vnnlib, seed, split_idx, nsplits, mask, strat
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', instance)
        db.commit()

def instance_already_verified(db, verifier, instance):
    cursor = db.cursor()
    query = '''
        SELECT 1 FROM verification_result WHERE
        verifier = ? AND
        problem = (
            SELECT id FROM instances WHERE
            benchmark = ? AND
            onnx = ? AND
            vnnlib = ? AND
            seed = ? AND
            split_idx = ? AND
            nsplits = ? AND
            mask = ? AND
            strat = ?
        )
    '''
    cursor.execute(query, (verifier, *instance))
    result = cursor.fetchone()
    return result is not None

def insert_into_verification_db(db, verifier, instance, status, time):
    cursor = db.cursor()
    cursor.execute('''
        INSERT INTO verification_result (
            verifier, status, time, problem
        ) VALUES (?, ?, ?, (
            SELECT id FROM instances WHERE
            benchmark = ? AND
            onnx = ? AND
            vnnlib = ? AND
            seed = ? AND
            split_idx = ? AND
            nsplits = ? AND
            mask = ? AND
            strat = ?
        ))
    ''', (verifier, status, time, *instance))
    db.commit()

def size_of_db(db, table="my_table"):
    cursor = db.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    return cursor.fetchone()[0]

def get_instances(benchmark):
    instances = []
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx_path, vnnlib_path, _ = line.strip().split(',')
            instances.append((onnx_path, vnnlib_path))
    return instances

def get_selected_instances(exp_home, benchmark):
    instances = []
    benchmark_name = benchmark["name"]
    with open(exp_home/f"selected_instances_{benchmark_name}.csv", "r") as f:
        # skip header
        f.readline()
        for line in f:
            cols = line.strip().split(',')
            onnx_path, vnnlib_path = cols[0], cols[1]
            instances.append((onnx_path, vnnlib_path))
    return instances

def run_splitter(onnx_path, vnnlib_path, output_dir, log_dir, split_idx, strat_n_mask, nsplits, seed, atol, rtol):
    wd = os.environ["TOOL_ROOT"]
    strat, mask = strat_n_mask
    fname = f"{onnx_path.stem}~{vnnlib_path.stem}~RS~{split_idx}~{mask}~{strat}~{nsplits}~{seed}"
    output_path = output_dir / f"{fname}.onnx"
    log_path  = log_dir / f"{fname}.log"

    cmd =   f"python main.py split --net {onnx_path} --spec {vnnlib_path} --output {output_path} "\
            f"--split_strategy {strat} --mask {mask} --split_idx {split_idx} "\
            f"--n_splits {nsplits} --seed {seed} --atol {atol} --rtol {rtol}"
    with open(log_path, "w") as f:
        ret = subprocess.run(cmd, shell=True, cwd=wd, stdout=f, stderr=f)
    return ret.returncode, fname


import os
import subprocess
import sqlite3

def get_db(db_root):
    db = sqlite3.connect(db_root)
    cursor = db.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        benchmark TEXT,
        onnx TEXT,
        vnnlib TEXT,
        split_idx INTEGER,
        split_strat TEXT,
        mask TEXT,
        nsplits INTEGER,
        seed INTEGER,
        atol REAL,
        rtol REAL,
        status TEXT,
        PRIMARY KEY (onnx, vnnlib, split_idx, split_strat, mask, nsplits, seed, atol, rtol)
    )
''')
    db.commit()
    return db

def size_of_db(db):
    cursor = db.cursor()
    cursor.execute('SELECT COUNT(*) FROM tasks')
    return cursor.fetchone()[0]


def get_instances(benchmark):
    instances = []
    with open(benchmark["path"] / "instances.csv", "r") as f:
        for line in f:
            onnx_path, vnnlib_path, _ = line.strip().split(',')
            instances.append((onnx_path, vnnlib_path))
    return instances

def already_in_db(db, benchmark, task):
    cursor = db.cursor()
    query = '''
        SELECT 1 FROM tasks WHERE
        benchmark = ? AND
        onnx = ? AND
        vnnlib = ? AND
        split_idx = ? AND
        split_strat = ? AND
        mask = ? AND
        nsplits = ? AND
        seed = ? AND
        atol = ? AND
        rtol = ?
    '''
    cursor.execute(query, (
        benchmark,
        str(task[0].stem),
        str(task[1].stem),
        task[4],
        task[5][0],
        task[5][1],
        task[6],
        task[7],
        task[8],
        task[9]
    ))
    result = cursor.fetchone()
    return result is not None


def insert_into_db(db, benchmark, task, val):
    if not already_in_db(db, benchmark, task):
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO tasks (
                benchmark, onnx, vnnlib, split_idx, split_strat, mask, nsplits, seed, atol, rtol, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            benchmark,
            str(task[0].stem),
            str(task[1].stem),
            task[4],
            task[5][0],
            task[5][1],
            task[6],
            task[7],
            task[8],
            task[9],
            val
        ))
        db.commit()

    
def run_splitter(onnx_path, vnnlib_path, output_dir, log_dir, split_idx, strat_n_mask, nsplits, seed, atol, rtol):
    wd = os.environ["TOOL_ROOT"]
    strat, mask = strat_n_mask
    fname = f"{onnx_path.stem}_{vnnlib_path.stem}_RS_{split_idx}_{mask}_{strat}_{nsplits}_{seed}"
    output_path = output_dir / f"{fname}.onnx"
    log_path  = log_dir / f"{fname}.log"

    cmd =   f"python main.py split --net {onnx_path} --spec {vnnlib_path} --output {output_path} "\
            f"--split_strategy {strat} --mask {mask} --split_idx {split_idx} "\
            f"--n_splits {nsplits} --seed {seed} --atol {atol} --rtol {rtol}"
    with open(log_path, "w") as f:
        subprocess.run(cmd, shell=True, cwd=wd, stdout=f, stderr=f)



# Database load and store functions 
# This file is meant to be %load'ed at the top of the
# various model-run notebooks


%load_ext sql

%config SqlMagic.autopandas = True 

%sql postgresql://localhost/bankcalls
    

import os.path
import pickle

from string import Template

def store_results(m):
    filename = m['name'] + '_' + m['strategy'] + '.pkl'
    dirname = '../data/inter/'
    pathname = dirname + filename
    
    count = 0
    while os.path.isfile(pathname):
        pathname = (dirname + 
                    m['name'] + 
                    '_' + 
                    m['strategy'] + 
                    str(count) +
                    '.pkl'
                   )
        count += 1
                    
    f = open(pathname, 'w+b')
    pickle.dump(m, f)
    f.close()
    
    # all the quotes and brackets seem to confuse %sql so I'm templating
    # the command manually
    sqlt = Template("""INSERT 
        INTO test_results(pathname, accuracy, recall, precision, 
                            f1, auc, cm_00, cm_01, cm_10, cm_11) 
        VALUES  ($pg_path, $accuracy, $recall, $precision, 
                            $f1, $auc, $cm_00, $cm_01, $cm_10, $cm_11);""")
    sqlins = sqlt.substitute(pg_path = "'" + pathname + "'",
                    accuracy = m['accuracy'],
                    recall = m['recall'],
                    precision = m['precision'],
                    f1 = m['f1'],
                    auc = m['auc'],
                    cm_00 = m['cm'][0,0],
                    cm_01 = m['cm'][0,1],
                    cm_10 = m['cm'][1,0],
                    cm_11 = m['cm'][1,1]
                   )
    %sql $sqlins
                    
    return pathname
    
        
def load_results(path):
    f = open(path, 'r+b')
    m = pickle.load(f)
    return m
from glob import glob
import os
import sys
# default settings
learners = ['afp','dc','ep-lex-dynamic','ep-lex-static','ep-lex-semidynamic','lex','randsel','tournament']
problems = ['uball5d',
            'tower',
            'enh',
            'enc',
            'housing',
            'airfoil',
            'concrete',
            'yacht'            
            ]

if len(sys.argv)>1:
    if sys.argv[1] == 'eplex':
        learners = ['ep-lex-dynamic','ep-lex-static','ep-lex-semidynamic']
    elif sys.argv[1] != 'all':
        learners = [sys.argv[1]]

if len(sys.argv)>2:
    if sys.argv[2] == 'regression':
        problems = ['uball5d',
                    'tower',
                    'enh',
                    'enc',
                    'housing',
                    'airfoil',
                    'concrete',
                    'yacht']
    elif sys.argv[2] != 'all':
        problems = [sys.argv[2]]

data_dir = '/home/lacava//data/regression/'
n_cores = 48
for p in problems:
    print(p)
    dataset_name = data_dir + 'd_' + p + '.txt'
    results_path = '/home/lacava/results/ep-lex-benchmark/trials/'
    for ml in learners:

        job_name = ml + '_' + p
        save_file = results_path + ml + '_' + p + '.txt'
        out_file = results_path + '{JOB_NAME}_%J.out'.format(JOB_NAME=job_name)
        error_file = out_file[:-4] + '.err'
        
        bjob_line = ('bsub -o {OUT_FILE} -e {ERROR_FILE} -n {N_CORES} -J {JOB_NAME} -R "span[hosts=1]" '
             '"python {ML}_trials.py {DATASET} {SAVE_FILE} {PROB} {SIZE}"'.format(OUT_FILE=out_file,
                                                    ERROR_FILE=error_file,
                                                    JOB_NAME=job_name,
                                                    N_CORES=n_cores,
                                                    ML=ml,
                                                    DATASET=dataset_name,
                                                    SAVE_FILE=save_file,
                                                    PROB=p,
                                                    SIZE=size))

        # print(bjob_line)
        os.system(bjob_line)

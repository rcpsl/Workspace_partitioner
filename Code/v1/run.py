from state_machine import preprocess as f
import multiprocessing
import numpy as np

def main():
    np.random.seed(0)
    cpu_cnt = multiprocessing.cpu_count()
    print('CPU count: %d'%cpu_cnt)

    #PARAMS

    from_region = [1,2]
    to_region = 23
    PREPROCESS = False
    USE_CTR_EX = False 
    max_iter = 30000
    verbosity = 'OFF'
    n_layers = 3
    start = 20
    step  = 10

    # layer_size = [start + step*i for i in range(4)]
    tasks_args = []
    for i in range(cpu_cnt - 2):
        layer_size = start + step*i
        fname = './experiments/exp_'+str(layer_size)+'_'+str(n_layers)
        tasks_args.append((from_region[0], from_region[1],to_region,PREPROCESS, 
                    USE_CTR_EX, max_iter,verbosity,n_layers,layer_size,fname))


    jobs = [multiprocessing.Process(target=f, args=args) for args in tasks_args]
    
    
    for job in jobs:
        job.daemon = True
        job.start()

    # Exit the completed processes
    for job in jobs:
        job.join()
    
    # preprocess(from_region[0], from_region[1],to_region,PREPROCESS, USE_CTR_EX, max_iter,verbosity,n_layers,layer_size)
def nth():
    print('Hi')
if __name__ == '__main__':
    main()

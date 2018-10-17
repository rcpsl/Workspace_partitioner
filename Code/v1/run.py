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
    PREPROCESS = True
    USE_CTR_EX = False 
    max_iter = 20000
    verbosity =False
    n_layers = 4
    start = 30
    step  = 10

    # layer_size = [start + step*i for i in range(4)]
    tasks_args = []
    for i in range(cpu_cnt):
        layer_size = start + step*i
        fname = './experiments/exp_'+str(layer_size)+'_'+str(n_layers)
        tasks_args.append((from_region[0], from_region[1],to_region,PREPROCESS, 
                    USE_CTR_EX, max_iter,verbosity,n_layers,layer_size,fname))


    print(tasks_args)    
    jobs = [multiprocessing.Process(target=f, args=args) for args in tasks_args]
    
    
    for job in jobs:
        job.start()

    # Exit the completed processes
    for job in jobs:
        job.join()
    
    # preprocess(from_region[0], from_region[1],to_region,PREPROCESS, USE_CTR_EX, max_iter,verbosity,n_layers,layer_size)
def nth():
    print('Hi')
if __name__ == '__main__':
    main()
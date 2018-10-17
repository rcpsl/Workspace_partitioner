from state_machine import preprocess as f
import multiprocessing
import numpy as np

def main():
    np.random.seed(0)
    cpu_cnt = multiprocessing.cpu_count()
    print('CPU count: %d'%cpu_cnt)

    #PARAMS

    from_region = [1,2]
    # to_region = 23
    to_regions = [0,2,3,6,7,13,23]
    PREPROCESS = False
    USE_CTR_EX = False 
    max_iter = 30000
    verbosity = 'OFF'
    n_layers = 3
    start = 10
    step  = 10

    # layer_size = [start + step*i for i in range(4)]
    # tasks_args = []
    # for i in range(cpu_cnt -1):
    #     layer_size = start + step*i
    #     if(PREPROCESS):
    #         fname = './experiments/exp_'+str(layer_size)+'_'+str(n_layers) +'_preprocess'
    #     elif(USE_CTR_EX):
    #         fname = './experiments/exp_'+str(layer_size)+'_'+str(n_layers) +'_CE'
    #     else:
    #         fname = './experiments/exp_'+str(layer_size)+'_'+str(n_layers)
 

    #     tasks_args.append((from_region[0], from_region[1],to_region,PREPROCESS, 
    #                 USE_CTR_EX, max_iter,verbosity,n_layers,layer_size,fname))

    tasks_args = []
    layer_size = 40
    for i in range(cpu_cnt -1):
        to_region = to_regions[i]
        if(PREPROCESS):
            fname = './experiments/3b/exp_'+str(to_region) +'_preprocess'
        elif(USE_CTR_EX):
            fname = './experiments/3b/exp_'+str(to_region)+'_CE'
        else:
            fname = './experiments/3b/exp_'+str(to_region) 

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

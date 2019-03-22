import pickle
from NN_verifier import *
from utility import *
import sys
import signal

def preprocess(frm_abst_index, frm_refined_index,to_abst_index,preprocess, use_ctr_examples, 
                max_iter,verbose,num_layers,hidden_layer_size,redirect_out):
    if(len(redirect_out) !=0):
        f = open(redirect_out, 'w')
        sys.stdout = f
    # Initiate workspace
    workspace  = Workspace()
    num_lasers = len(workspace.laser_angles)

    # Initiate trained NN
    trained_nn = NeuralNetworkStruct(num_lasers,num_layers,hidden_layer_size, load_weights=True)

    # Load regions
    # filename = './regions/regions10-15/abst_reg_H_rep_with_obstacles.txt'
    # with open(filename, 'rb') as inputFile:
    #     abst_reg_H_rep_with_obstacles = pickle.load(inputFile)

    filename = './regions/regions10-15/abst_reg_H_rep.txt'
    with open(filename, 'rb') as inputFile:
        abst_reg_H_rep = pickle.load(inputFile)

    filename = './regions/regions10-15/abst_reg_V_rep.txt'
    with open(filename, 'rb') as inputFile:
        abst_reg_V_rep = pickle.load(inputFile)

    filename = './regions/regions10-15/refined_reg_H_rep_dict.txt'
    with open(filename, 'rb') as inputFile:
        refined_reg_H_rep_dict = pickle.load(inputFile)

    filename = './regions/regions10-15/refined_reg_V_rep_dict.txt'
    with open(filename, 'rb') as inputFile:
        refined_reg_V_rep_dict = pickle.load(inputFile)

    filename = './regions/regions10-15/lidar_config_dict.txt'
    with open(filename, 'rb') as inputFile:
        lidar_config_dict = pickle.load(inputFile)

    """
    print len(abst_reg_H_rep_with_obstacles)    
    print len(abst_reg_H_rep) 
    print len(abst_reg_V_rep) 
    print len(refined_reg_H_rep_dict)
    print len(refined_reg_V_rep_dict)
    print len(lidar_config_dict)
    """
    """
    for abst_index, this_abst_reg_V in enumerate(abst_reg_V_rep):
        print '\n\n abst_index = ', abst_index
        print_region(this_abst_reg_V)
    """

    frm_pt = np.array([1,0.5]) 
    to_pt = np.array([5.430652751587331, 6.135549735277891])
    
    frm_abst_index = in_region(abst_reg_H_rep,frm_pt)
    to_abst_index  = in_region(abst_reg_H_rep,to_pt)
    frm_abst_index = 2
    to_abst_index =5
    print 'From Regions %d, to region %d'%(frm_abst_index,to_abst_index)
    frm_lidar_config  = lidar_config_dict[frm_abst_index]


    # frm_refined_reg_H = refined_reg_H_rep_dict[frm_abst_index][frm_refined_index]
    # frm_refined_reg_V = refined_reg_V_rep_dict[frm_abst_index][frm_refined_index] 

    frm_abst_reg_H = abst_reg_H_rep[frm_abst_index]
    to_abst_reg_H = abst_reg_H_rep[to_abst_index]
    #to_abst_reg_V = abst_reg_V_rep[to_abst_index]

    # print 'frm_refined_reg_V = '
    # print_region(frm_refined_reg_V)

    #print '\n\nto_abst_reg_V = '
    #print_region(to_abst_reg_V)
    #print '\n\nto_abst_reg_H = ', to_abst_reg_H

    # print '\n\nfrm_lidar_config = ', frm_lidar_config


    # Initialize VerifyNNParser
    num_integrators = 2
    Ts = 0.5
    higher_deriv_bound = 2.0
    parser = NN_verifier(trained_nn, workspace, num_integrators, Ts, higher_deriv_bound,out_file)

    parser.parse(frm_abst_reg_H, to_abst_reg_H, frm_lidar_config, frm_abst_index, frm_abst_index,preprocess, use_ctr_examples, max_iter,verbose)


def create_cmd_parser():
    arg_parser = argparse.ArgumentParser(description='Input arguments)')

    arg_parser.add_argument('L', help = 'Number of layers including the output')
    arg_parser.add_argument('H', help = 'Number of neurons of hidden layers')
    arg_parser.add_argument('from_R', nargs = 2, help = 'Start region,(abstract index)(refined index)')
    arg_parser.add_argument('to_R', nargs = 1 , help = 'End region,(abstract index)(refined index)')
    arg_parser.add_argument('preprocess',default = True , help = "Preprocessing flag,default is True")
    arg_parser.add_argument('--use_ctr_examples',default = True , help = "Use counter examples when not pre-processing")
    arg_parser.add_argument('--max_iter', default = 100000, help ="Solver max iterations")
    arg_parser.add_argument('--verbosity', default = 'OFF', help ="Solver Verbosity")
    arg_parser.add_argument('--load_weights', default = False, help ="Load weights, layer size must be 200")
    arg_parser.add_argument('--abs_goal', default = False, help ="1 if goal is an abstract region")
    arg_parser.add_argument('--file', default = '', help ="Output file")

    return arg_parser

def handler(signum, frame):
    print("Time out")
    raise Exception("TLE")

def in_region(regions,x):
    
    ret = -1
    for idx,region in enumerate(regions):
        H = np.array(region['A'])
        b = np.array(region['b'])
        if(np.less_equal(H.dot(x),b).all()):
            if(ret != -1):
              return -2
            else:
                ret = idx
    return ret
    
if __name__ == '__main__':
    arg_parser = create_cmd_parser()
    ns = arg_parser.parse_args()
    layer_size = int(ns.H)
    n_hidden_layers = int(ns.L)
    from_region = [int(i) for i in ns.from_R]
    to_region = [int(i) for i in ns.to_R]
    PREPROCESS = bool(int(ns.preprocess))
    USE_CTR_EX = bool(int(ns.use_ctr_examples))
    max_iter = int(ns.max_iter)
    load_weights = bool(int(ns.load_weights))
    abs_goal = bool(int(ns.abs_goal))
    verbosity = ns.verbosity
    out_file = ns.file
     
    print('=========================================')
    print('NN hidden layer size:', layer_size)
    print('From Region (%d,%d)' %(from_region [0],from_region[1]))
    print('To Region (%d)' %to_region[0])
    print('Preprocess: %s' %PREPROCESS)
    print('Use_counter_examples: %s '%USE_CTR_EX)
    print('Solver max iterations: %d' %max_iter)
    # print('load_weights: %s' %load_weights)
    # print('Abstract goal: %s' %abs_goal)

    np.random.seed(0)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(3600*5)
    preprocess(from_region[0], from_region[1],to_region[0],PREPROCESS, USE_CTR_EX, max_iter,verbosity,n_hidden_layers,layer_size,[])

    # try:
        # preprocess(from_region[0], from_region[1],to_region[0],PREPROCESS, USE_CTR_EX, max_iter,verbosity,n_hidden_layers,layer_size,[])
    # except Exception, exc: 
    #     print(exc)
    #     if(len(out_file) > 0):
    #         f = open(out_file, 'a+')
    #         f.write('\t\t\tTLE')
    #         f.close()

    print('=========================================')

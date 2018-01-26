import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, PEPGVariant
from testbed import * 
import pickle 




def config():
	global NPARAMS
	global NPOPULATION
	global MAX_ITERATION
	global fit_func

	NPARAMS = 2        # make this a 100-dimensinal problem.
	NPOPULATION = 101    # use population size of 101.
	MAX_ITERATION = 4000 # run each solver for 5000 generations.
	# fit_func=rastrigin
	# fit_func = dejong
	# fit_func = hyperEllipsoid
	# fit_func =schwefel
	# fit_func =griewangk
	fit_func = rosebrock

	NPOPULATION=int(4+3*np.ceil(np.log(NPOPULATION)))
	print(NPOPULATION)
	np.random.seed(0)
	
	

def test_solver(solver):
	history = []
	for j in range(MAX_ITERATION):
		solutions = solver.ask()
		fitness_list = np.zeros(solver.popsize)
		for i in range(solver.popsize):
			fitness_list[i] = fit_func(solutions[i])
		solver.tell(fitness_list)
		result = solver.result() # first element is the best solution, second element is the best fitness
		history.append(abs(result[1]))
		if (j+1) % 100 == 0:
		  print("fitness at iteration", (j+1), result[1])
	# print("local optimum discovered by solver:\n", result[0])
	print("fitness score at this local optimum:", result[1])
	return history




def debug_solver(solver):

	history_population=[] 

	for j in range(MAX_ITERATION):
		solutions =solver.ask()
		fitness_list = np.zeros(solver.popsize)
		for i in range(solver.popsize):
			fitness_list[i]=fit_func(solutions[i])
		solver.tell(fitness_list)
		# result =solver.result()
		if (j+1)%100 ==0:
			print("fitness at iteration",(j+1),fitness_list.mean())

		history_population.append(solutions)

	return history_population 

def testRun():
	config()
	x = np.random.randn(NPARAMS)
	print("The fitness of initial guess", fit_func(x)) 

	pepg = PEPG(NPARAMS,                         # number of model parameters
	    sigma_init=0.5,                  # initial standard deviation
	    learning_rate=0.1,               # learning rate for standard deviation
	    learning_rate_decay=1.0,       # don't anneal the learning rate
	    popsize=NPOPULATION,             # population size
	    average_baseline=False,          # set baseline to average of batch
	    weight_decay=0.00,            # weight decay coefficient
	    rank_fitness=False,           # use rank rather than fitness numbers
	    forget_best=False)            # don't keep the historical best solution)

	pepg_history = test_solver(pepg)  #


	pepgV = PEPGVariant(NPARAMS,                         # number of model parameters
    sigma_init=0.5,                  # initial standard deviation
    learning_rate=0.1,               # learning rate for standard deviation
    learning_rate_decay=1.0,       # don't anneal the learning rate
    popsize=NPOPULATION,             # population size
    average_baseline=False,          # set baseline to average of batch
    weight_decay=0.00,            # weight decay coefficient
    rank_fitness=False,           # use rank rather than fitness numbers
    forget_best=False,            # don't keep the historical best solution
    diversity_best=0.1)            # use the diversity issue for just testing 

	print("-----test PEPG vairant-----")
	pepgv_history = test_solver(pepgV)  #


	print("---test PEPG variant with different diversity-----")

	pepgV2 = PEPGVariant(NPARAMS,                         # number of model parameters
    sigma_init=0.5,                  # initial standard deviation
    learning_rate=0.1,               # learning rate for standard deviation
    learning_rate_decay=1.0,       # don't anneal the learning rate
    popsize=NPOPULATION,             # population size
    average_baseline=False,          # set baseline to average of batch
    weight_decay=0.00,            # weight decay coefficient
    rank_fitness=False,           # use rank rather than fitness numbers
    forget_best=False,            # don't keep the historical best solution
    diversity_best=1)            # use the diversity issue for just testing 


	# done 

	pepgV2_history =test_solver(pepgV2)



	
	oes = OpenES(NPARAMS,                  # number of model parameters
	    sigma_init=0.5,            # initial standard deviation
	    sigma_decay=0.999,         # don't anneal standard deviation
	    learning_rate=0.1,         # learning rate for standard deviation
	    learning_rate_decay = 1.0, # annealing the learning rate
	    popsize=NPOPULATION,       # population size
	    antithetic=False,          # whether to use antithetic sampling
	    weight_decay=0.00,         # weight decay coefficient
	    rank_fitness=False,        # use rank rather than fitness numbers
	    forget_best=False)

	print("-----test oes--------------")
	oes_history = test_solver(oes)


	cmaes = CMAES(NPARAMS,
	          popsize=NPOPULATION,
	          weight_decay=0.0,
	          sigma_init = 0.5
	      )
	cma_history = test_solver(cmaes)


	best_history = [0] * MAX_ITERATION
	plt.figure(figsize=(16,8), dpi=150)

	optimum_line, = plt.plot(best_history, color="black", linewidth=0.5, linestyle="-.", label='Global Optimum')
	pepgv_line, = plt.plot(pepgv_history, color="red", linewidth=1.0, linestyle="-", label='PEPGV / NES')
	pepg_line, = plt.plot(pepg_history, color="blue", linewidth=1.0, linestyle="-.", label='PEPG / NES')
	oes_line, = plt.plot(oes_history, color="orange", linewidth=1.0, linestyle="-", label='OpenAI-ES')
	cma_line, = plt.plot(cma_history, color="green", linewidth=1.0, linestyle="-", label='CMA-ES')
	
	


	plt.legend(handles=[optimum_line,pepgv_line,pepg_line, oes_line], loc='best')


	plt.xlim(0,100)

	plt.xlabel('generation')
	plt.ylabel('loss')

	plt.savefig("./results/rose_"+str(NPARAMS)+"d.svg")





def debugRun():

	config()
	x = np.random.randn(NPARAMS)
	print("The fitness of initial guess", fit_func(x)) 

	# oes = OpenES(NPARAMS,                  # number of model parameters
	#     sigma_init=0.5,            # initial standard deviation
	#     sigma_decay=0.999,         # don't anneal standard deviation
	#     learning_rate=0.1,         # learning rate for standard deviation
	#     learning_rate_decay = 1.0, # annealing the learning rate
	#     popsize=NPOPULATION,       # population size
	#     antithetic=False,          # whether to use antithetic sampling
	#     weight_decay=0.00,         # weight decay coefficient
	#     rank_fitness=False,        # use rank rather than fitness numbers
	#     forget_best=False)

	# print("-----test oes--------------")
	pepg = PEPG(NPARAMS,                         # number of model parameters
    sigma_init=0.5,                  # initial standard deviation
    learning_rate=0.1,               # learning rate for standard deviation
    learning_rate_decay=1.0,       # don't anneal the learning rate
    popsize=NPOPULATION,             # population size
    average_baseline=False,          # set baseline to average of batch
    weight_decay=0.00,            # weight decay coefficient
    rank_fitness=False,           # use rank rather than fitness numbers
    forget_best=False)            # don't keep the historical best solution)

	# pepg_history = test_solver(pepg)  #

	history = debug_solver(pepg)

	history=np.array(history)

	print(history.shape) # done 


	pickle_out=open("pepg_rose.pickle","wb")

	pickle.dump(history,pickle_out)

	pickle_out.close()








	



if __name__=="__main__":
	testRun()
	debugRun()




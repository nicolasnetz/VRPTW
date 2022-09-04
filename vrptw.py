import pandas as pd
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
import time

class Instance():
    def __init__(self,file_path, num_nodos = None):
        # Instance characteristics
        self.file_path = file_path
        self.ins_name = self.file_path.split(sep = '/')[1].split('.')[0]
        self.num_nodos = num_nodos
        # Choose the number of nodes
        if num_nodos != None:
            self.ins_name = "{}-{}".format(self.ins_name,str(num_nodos))
    
    def get_data(self):
        # Retrieve data
        instance_data = []
        with open(self.file_path) as f:
            file = f.readlines()
            self.vehicle_number = int(file[4].replace("\n","").split("\t")[0].split(" ")[2])
            self.vehicle_capacity = int(file[4].replace("\n","").split("\t")[0].split(" ")[-1])

            if self.num_nodos == None:
                for line in file[9:]:
                    line_data = [int(i) for i in line.split(" ") if i not in ["", "\n"]]
                    instance_data.append(line_data)
            if self.num_nodos!= None:
                for line in file[9:10+self.num_nodos]:
                    line_data = [int(i) for i in line.split(" ") if i not in ["", "\n"]]
                    instance_data.append(line_data)

            # Include n + 1
            n_1 = instance_data[0].copy()
            n_1[0] = len(instance_data)
            instance_data.append(n_1)

        # Take columns
        instance_data = np.array(instance_data).astype(int)
        self.cust_no = instance_data[:,0]
        self.coord_x = instance_data[:,1]
        self.coord_y = instance_data[:,2]
        self.demand = instance_data[:,3]
        self.ready_time = instance_data[:,4]
        self.due_date = instance_data[:,5]
        self.service_time = instance_data[:,6]

        # Mathematical model data
        self.V = list(self.cust_no)
        self.N = list(self.cust_no[1:-1]) # V\{0, n + 1}
        self.K = [i for i in range(self.vehicle_number)]
        self.A = [(i,j) for i in self.V for j in self.V if i not in [j,self.V[-1]] if j!=0]
        self.c = {(i,j): int(10*np.sqrt( (self.coord_x[i] - self.coord_x[j] )**2 + (self.coord_y[i] - self.coord_y[j])**2))/10 for i,j in self.A}
        self.a = list(self.ready_time.copy())
        self.b = list(self.due_date.copy())
        self.s = list(self.service_time.copy())
        self.Q = self.vehicle_capacity
        self.q = list(self.demand.copy())
        self.M = {(i,j): np.max(self.b[i]+self.s[i]+self.c[i,j]-self.a[j],0) for i,j in self.A}

class Vrptw():
    def __init__(self,log_output = True, time_limit = 30):
        self.name = "VRPTW1 from Ch.5 Toth&Vigo, 2014"
        self.log_output = log_output
        self.time_limit = time_limit

    def build(self,ins):
        # Model
        self.mdl = Model(self.name)

        # Decision Variables
        self.x = self.mdl.binary_var_dict([(i,j,k) for i,j in ins.A for k in ins.K], name='x')
        self.T = self.mdl.continuous_var_dict([(i,k) for i in ins.V for k in ins.K],name='T')

        # Constraint 1
        self.mdl.minimize(self.mdl.sum(self.x[(i,j,k)]*ins.c[(i,j)] for i,j,k in [(i,j,k) for i,j in ins.A for k in ins.K]))
        # Constraint 2
        self.mdl.add_constraints(self.mdl.sum(self.x[(i,j,k)] for j in ins.V if j not in [0,i] for k in ins.K)==1 for i in ins.N)
        # Constraint 3
        self.mdl.add_constraints(self.mdl.sum(self.x[0,j,k] for j in ins.V if j!=0)==1 for k in ins.K)
        # Constraint 4        
        self.mdl.add_constraints(self.mdl.sum(self.x[(i,j,k)] for i in ins.V if i not in [j,ins.V[-1]]) == self.mdl.sum(self.x[j,i,k] for i in ins.V if i not in [j,0]) for j in ins.N for k in ins.K)
        # Constraint 5        
        self.mdl.add_constraints(self.mdl.sum(self.x[i,ins.V[-1],k] for i in ins.V if i!=ins.V[-1]) == 1 for k in ins.K)
        # Constraint 6        
        self.mdl.add_constraints(self.T[i,k] + ins.c[i,j] + ins.s[i] -ins.M[i,j]*(1 - self.x[(i,j,k)]) <= self.T[j,k] for i,j in ins.A for k in ins.K)
        # Constraint 7        
        self.mdl.add_constraints(self.T[i,k] >= ins.a[i] for i in ins.V for k in ins.K )
        # Constraint 8        
        self.mdl.add_constraints(self.T[i,k] <= ins.b[i] for i in ins.V for k in ins.K )
        # Constraint 9       
        self.mdl.add_constraints((self.mdl.sum(ins.q[i]*(self.mdl.sum(self.x[(i,j,k)] for j in ins.V if j not in [0,i])) for i in ins.N) <= ins.Q) for k in ins.K)
   
    def solve(self,ins):
        print("Solving instance {}".format(ins.ins_name))
        self.start_time = time.time()
        self.mdl.parameters.timelimit= self.time_limit
        self.mdl.context.cplex_parameters.threads = 4
        self.solution = self.mdl.solve(log_output=self.log_output)
        self.finish_time = time.time()
        self.elapsed_time = self.finish_time - self.start_time
        self.result_type = str(self.mdl.solve_status).split(".")[1]
        if self.solution:
            print("Objective: {0:.1f}".format(self.mdl.objective_value))
            print("Result: ", self.result_type)
            
        else: 
            print("Model for {} has no solution...".format(ins.ins_name))

    def print_sol(self):
        self.mdl.report()
        print(self.mdl.solution)

    def export_mdl_file(self,ins):
        # print(self.mdl.pprint_as_string())         
        self.mdl.export_as_lp("my_models/model{}.lp".format(ins.ins_name))

    def plot_sol(self,ins,show_plot = False):
        routes = []
        truck = []
        for k in ins.K:
            for i in ins.V:
                if i !=0 and self.x[0,i,k].solution_value >0.9:
                    aux = [0,i]
                    while i != 0 and i != (len(ins.V) - 1):
                        j = i
                        for h in ins.V:
                            try:
                                if j !=h and self.x[j,h,k].solution_value > 0.9:
                                    aux.append(h)
                                    i = h
                            except:
                                pass    
                    routes.append(aux)
                    truck.append(k)


        routes_by_vehicle = []
        for i in truck:
            tuples = []
            for j in range(len(routes[i])):
                try:
                    tuples.append((routes[i][j],routes[i][j + 1]))
                except:
                    pass
            routes_by_vehicle.append(tuples)

        plt.figure(figsize = (12,5))
        plt.scatter(ins.coord_x,ins.coord_y, color = "blue")

        # DC
        plt.scatter(ins.coord_x[0],ins.coord_y[0], color = "red", marker = "D")


        color = iter(cm.rainbow(np.linspace(0, 1, ins.vehicle_number)))
        for v in truck:
            c = next(color)
            for i,j in routes_by_vehicle[v]:
                plt.plot([ins.coord_x[i],ins.coord_x[j]],[ins.coord_y[i],ins.coord_y[j]], alpha=0.4, zorder=0, c = c)

        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Solution of \n {0} \n {1:.1f}\n {2}".format(ins.ins_name,self.mdl.objective_value, self.result_type))
        if show_plot:
            plt.show()
        else:    
            plt.savefig('my_plots/{}.png'.format(ins.ins_name))

    def export_sol_file(self, ins):
        routes = []
        truck = []
        for k in ins.K:
            for i in ins.V:
                if i !=0 and self.x[0,i,k].solution_value >0.9:
                    aux = [0,i]
                    while i != 0 and i != (len(ins.V) - 1):
                        j = i
                        for h in ins.V:
                            try:
                                if j !=h and self.x[j,h,k].solution_value > 0.9:
                                    aux.append(h)
                                    i = h
                            except:
                                pass    
                    routes.append(aux)
                    truck.append(k)


        routes_by_vehicle = []
        for i in truck:
            tuples = []
            for j in range(len(routes[i])):
                try:
                    tuples.append((routes[i][j],routes[i][j + 1]))
                except:
                    pass
            routes_by_vehicle.append(tuples)

        formatted_routes = []
        for route in routes_by_vehicle:
            if len(route) == 1:
                continue
            else:
                aux = [0]
                for path in route:
                    if path[0] in aux:
                        continue
                    else:
                        if path[1] == 101:
                            continue
                        else:
                            aux.append(path[0])
                            aux.append(path[1])
                aux = aux[1:]
                formatted_routes.append(aux)

        with open('my_solutions/{}.sol'.format(ins.ins_name), 'w') as fp:
            fp.write("{}\n".format(ins.ins_name))
            for i in range(1, len(formatted_routes) + 1):
                fp.write("Route: #{}: {}\n".format(i, formatted_routes[i - 1]) )
                
            fp.write("{}\n".format(self.mdl.objective_value))

    def export_to_table(self,ins):
        if self.solution:
            return [ins.ins_name, ins.num_nodos, "{0:.1f}".format(self.mdl.objective_value), self.result_type,  "{0:.1f}".format(self.mdl.solve_details.mip_relative_gap * 100), "{0:.4g}".format(self.elapsed_time)]
        else: 
            return [ins.ins_name, ins.num_nodos, "DNF", self.result_type, "NaN", "{0:.4g}".format(self.elapsed_time)]

class Solver():
    def __init__(self, file_path = 'instances/C101.txt', num_nodos = 10, log_output = True, time_limit = 30, show_plot = False, plot = False ):
        self.file_path = file_path # 
        self.num_nodos = num_nodos #
        self.log_output = log_output #
        self.time_limit = time_limit #
        self.show_plot = show_plot
        self.plot = plot

        self.ins = Instance(file_path = self.file_path, num_nodos=self.num_nodos)
        self.model = Vrptw(log_output=log_output, time_limit=self.time_limit)

    def execute(self):
        # Get instance data
        self.ins.get_data()
        # Build model
        self.model.build(self.ins)
        # Solve model
        self.model.solve(self.ins)
        # Exports
        self.model.export_mdl_file(self.ins)
        self.model.export_sol_file(self.ins)
        ## Plots
        if self.plot:
            self.model.plot_sol(ins=self.ins, show_plot= self.show_plot)
        
    def to_table(self):
        df = pd.read_csv('my_results.csv', usecols = [i for i in range(1,6 + 1)])
        df = df.append(dict(zip(df.columns,self.model.export_to_table(self.ins))), ignore_index = True)
        df.to_csv('my_results.csv')
   
if __name__=="__main__":
    try: 
        os.mkdir("my_solutions")
    except:
        pass
    try:
        os.mkdir("my_plots")
    except:
        pass
    try:
        os.mkdir("my_models")
    except:
        pass
    try:
        df = pd.read_csv('my_results.csv')
        df = df.append(dict(zip(df.columns,['new execution' for i in range(6)])), ignore_index = True)
        df.to_csv('my_results.csv')
    except:
        df = pd.DataFrame(columns=["instance", "nodes", "result", "status","gap %", "time"])
        df.to_csv("my_results.csv")

    # Instances paths
    instance_path = sorted(['instances/' + i for i in os.listdir('instances')])

    # For each file solve it 
    for ins_file_path in instance_path:
        s = Solver(
            file_path = ins_file_path,
            num_nodos = 10,
            log_output = False,
            time_limit = 30,
            show_plot = False,
            plot = True
            )
        s.execute()
        s.to_table()    

# Work to do
# Include time of execution to table?
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import sys

class Instancia():
    def __init__(self,file_path, include_n_1 = False, num_nodos = None):
        self.file_path = file_path
        file_head = []
        cols_head = []
        instance_data = []
        with open(file_path) as f:
            file = f.readlines()
            file_name = file[0]
            self.vehicle_number = int(file[4].replace("\n","").split("\t")[0].split(" ")[2])
            self.vehicle_capacity = int(file[4].replace("\n","").split("\t")[0].split(" ")[-1])

            if num_nodos == None:
                for line in file[9:]:
                    line_data = [int(i) for i in line.split(" ") if i not in ["", "\n"]]
                    instance_data.append(line_data)
            if num_nodos!= None:
                for line in file[9:10+num_nodos]:
                    line_data = [int(i) for i in line.split(" ") if i not in ["", "\n"]]
                    instance_data.append(line_data)

            if include_n_1 == True:
                # Include n + 1
                n_1 = instance_data[0].copy()
                n_1[0] = len(instance_data)
                instance_data.append(n_1)

        instance_data = np.array(instance_data).astype(int)
        self.cust_no = instance_data[:,0]
        self.coord_x = instance_data[:,1]
        self.coord_y = instance_data[:,2]
        self.demand = instance_data[:,3]
        self.ready_time = instance_data[:,4]
        self.due_date = instance_data[:,5]
        self.service_time = instance_data[:,6]

        # Mathematical model data
        self.V = list(self.cust_no) # All nodes
        if include_n_1 == True:
            self.N = list(self.cust_no[1:-1]) # V\{0, n + 1}            
        else:
            self.N = list(self.cust_no[1:]) # V\{0}    

        self.K = [i for i in range(self.vehicle_number)] # All vehicles
        self.A = [(i,j) for i in self.V for j in self.V if i not in [j,self.V[-1]] if j!=0]

        # Toth y Vigo
        self.c = {(i,j): np.hypot(self.coord_x[i] - self.coord_x[j] , self.coord_y[i] - self.coord_y[j]) for i,j in self.A}
        # Solomon
        # self.c = {(i,j): np.round(np.hypot(self.coord_x[i] - self.coord_x[j] , self.coord_y[i] - self.coord_y[j]),1) for i in self.V for j in self.V if i!=j}
        self.a = list(self.ready_time.copy())
        self.b = list(self.due_date.copy())
        self.s = list(self.service_time.copy())
        self.Q = self.vehicle_capacity
        self.q = list(self.demand.copy())
        self.M = {(i,j): np.max(self.b[i]+self.s[i]+self.c[i,j]-self.a[j],0) for i,j in self.A}


class Modelo():
    def __init__(self):
        self.name = "VRPTW1"

    def build(self,ins):
        # Model
        # Model is VRPTW1 from Ch.5 Toth&Vigo, 2014
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

        # print(self.mdl.pprint_as_string())         
        self.mdl.parameters.timelimit=3600
        self.mdl.context.cplex_parameters.threads = 1
        self.mdl.export_as_lp("model_nico.lp")


    def build(self,ins):
        # Model
        # Model is VRPTW1 from Ch.5 Toth&Vigo, 2014
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

        # print(self.mdl.pprint_as_string())         
        self.mdl.parameters.timelimit=3600
        self.mdl.context.cplex_parameters.threads = 1
        self.mdl.export_as_lp("model_nico.lp")

    def buildSolomon(self,ins):
        # Model
        # Model is SOlomon (1983)
        self.mdl = Model(self.name)

        # Decision Variables
        self.x = self.mdl.binary_var_dict([(i,j,k) for i in ins.V for j in ins.V for k in ins.K if i!=j], name='x')
        self.y = self.mdl.continuous_var_dict([(i,k) for i in ins.V for k in ins.K],name='y')
        self.t = self.mdl.continuous_var_dict([i for i in ins.V],name='t')
        self.t_0_s = self.mdl.continuous_var_dict([k for k in ins.K], name='t_0_s')
        self.t_0_f = self.mdl.continuous_var_dict([k for k in ins.K], name='t_0_f')


        # Constraint 1
        self.mdl.minimize(self.mdl.sum(self.x[(i,j,k)]*ins.c[(i,j)] for i,j,k in [(i,j,k) for i in ins.V for j in ins.V for k in ins.K if i!=j]))

        # Constraint 2
        self.mdl.add_constraints((self.mdl.sum(ins.q[i]*self.y[i,k] for i in ins.V)<= ins.Q for k in ins.K))

        # Constraint 3
        self.mdl.add_constraints(self.mdl.sum(self.y[i,k] for k in ins.K) == 1 for i in ins.N)
        self.mdl.add_constraint(self.mdl.sum(self.y[0,k] for k in ins.K)== ins.vehicle_number)

        # Constraint 4
        self.mdl.add_constraints(self.mdl.sum(self.x[(i,j,k)] for i in ins.V if i!=j) == self.y[j,k] for j in ins.V for k in ins.K)

        # Constraint 5
        self.mdl.add_constraints(self.mdl.sum(self.x[(i,j,k)] for j in ins.V if i!=j) == self.y[i,k] for i in ins.V for k in ins.K)

        # Constraint 6
        self.mdl.add_constraints(self.t[i] + ins.c[i,j] + ins.s[i] - 10000000*(1 - self.x[(i,j,k)]) <= self.t[j] for i in ins.N for j in ins.N for k in ins.K if i!=j)

        # Constraint 7
        self.mdl.add_constraints(self.t_0_f[k] >= self.t[i] + ins.s[i] + ins.c[i,0] -10000000*(1 - self.x[(i,0,k)]) for i in ins.N for k in ins.K)

        # Constraint 8
        self.mdl.add_constraints(self.t[j] >= self.t_0_s[k] + ins.s[j] + ins.c[0,j] - 1000000*(1 - self.x[(0,j,k)]) for j in ins.N for k in ins.K)

        # Constraint 9       
        self.mdl.add_constraints(self.t[i] >= ins.a[i] for i in ins.N)        
        self.mdl.add_constraints(self.t[i] <= ins.b[i] for i in ins.N)

        # Constraint 10
        self.mdl.add_constraints(self.t_0_s[k] >= ins.a[0] for k in ins.K)        
        self.mdl.add_constraints(self.t_0_s[k] <= ins.b[0] for k in ins.K)
        self.mdl.add_constraints(self.t_0_f[k] >= ins.a[0] for k in ins.K)        
        self.mdl.add_constraints(self.t_0_f[k] <= ins.b[0] for k in ins.K)


        # print(self.mdl.pprint_as_string())         
        self.mdl.parameters.timelimit=3600
        self.mdl.context.cplex_parameters.threads = 1
        self.mdl.export_as_lp("model_nico_solomon.lp")


    def solve(self):
        solution = self.mdl.solve(log_output=True)

        if solution:
            print("Obj: ",self.mdl.objective_value)
            self.mdl.report()
            
        else: 
            print("model has no solution...")

    def export_sol(self,ins):
        print(self.mdl.solution)

    def graficar_solucion(self,ins):
        rutas = []
        truck = []

        for k in ins.K:
            for i in ins.V:
                if i != 0 and self.x[0,i,k].solution_value >0.9:
                    aux = [0,i]
                    while i != 0:
                        j = i
                        for h in ins.V:
                            if j !=h and self.x[j,h,k].solution_value > 0.9:
                                aux.append(h)
                                i = h
                                
                    rutas.append(aux)
                    truck.append(k)


        caminos_por_truck = []
        for i in truck:
            tuplas = []
            for j in range(len(rutas[i])):
                try:
                    tuplas.append((rutas[i][j],rutas[i][j + 1]))
                except:
                    pass
            caminos_por_truck.append(tuplas)

        plt.figure(figsize = (12,5))
        plt.scatter(ins.coord_x,ins.coord_y, color = "blue")

        # DC
        plt.scatter(ins.coord_x[0],ins.coord_y[0], color = "red", marker = "D")


        color = iter(cm.rainbow(np.linspace(0, 1, ins.vehicle_number)))
        for v in truck:
            c = next(color)
            for i,j in caminos_por_truck[v]:
                plt.plot([ins.coord_x[i],ins.coord_x[j]],[ins.coord_y[i],ins.coord_y[j]], alpha=0.4, zorder=0, c = c)

        plt.xlabel("Distancia X")
        plt.ylabel("Distancia Y")
        plt.title("Solución {}".format(ins.file_path))
        plt.legend()
        plt.show()




if __name__=="__main__":
    # Ejemplo de entrada
    # python main.py C101 10
    print(sys.argv)
    ins_name = sys.argv[1]
    num_nodos = int(sys.argv[2])    
    ins = Instancia("instances/{}.txt".format(ins_name),num_nodos=num_nodos, include_n_1=True) # FALSE FOR SOLOMON
    model = Modelo()
    print("building model")
    model.build(ins)
    # model.buildSolomon(ins)
    # print(ins.V)
    # print(ins.N)
    print("model built")
    model.solve()
    # model.graficar_solucion(ins)


# Mejorar el origen de los indices, para no generar el objeto iterador cada vez

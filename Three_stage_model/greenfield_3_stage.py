import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
from mipcl_py.mipshell.mipshell import *
from networkx.algorithms import matching
#plt.style.use('ggplot')

class Facility_Location(Problem):
    def model(self, demand, supply, max_capacity, transportation_cost_pd, \
              transportation_cost_dc, operating_cost):
        """ Input parameters:
            D: list of size LxI, where D[l][i] is the demand for product l at customer location i
            S: list of size LxK, where S[l][k] is the supply of product l at plant k
            Umax: list of size J, where Umax[j] is the maximum throughput of DC at candidate location j
            cpd: list of size LxKxJ, where cpd[l][k][j] is the unit cost of shipping product l from plant k to candidate location j
            cdc: list of size LxJxI, where cdc[l][j][i] is the unit cost of shipping product l from candidate location j to customer i
            f: list of size J, where f[j] is fixed operating cost of a DC at site j
        """
        
        self.D = D = np.transpose(demand.values)
        self.S = S = np.transpose(supply.values)
        self.Umax = Umax = np.array(max_capacity.values.flatten())
        self.f = f = operating_cost.values.flatten()
        No_of_products = len(D)
        cpd1 = np.tile(transportation_cost_pd, (No_of_products, 1, 1))
        # np.tile does the repeatition for each product
        self.cpd = cpd = np.true_divide(cpd1, 5)
        cdc1 = np.tile(transportation_cost_dc, (No_of_products, 1, 1))
        # np.tile does the repeatition for each product
        self.cdc = cdc = np.true_divide(cdc1, 5)
        
        self.I, self.J, self.K, self.L = len(D[0]), len(f), len(S[0]), len(D)
        self.x = x = VarVector([self.J],'x', BIN)
        self.zpd = zpd = VarVector([self.L, self.K, self.J],'zpd', INT, lb = 0.0)
        self.zdc = zdc = VarVector([self.L, self.J, self.I],'zdc', INT, lb = 0.0)
	
        self.demand = demand
        self.supply = supply
        self.operating_cost = operating_cost
        
        minimize(
            sum_(f[j]*x[j] for j in range(self.J)) + \
            sum_(cpd[l][k][j]*zpd[l][k][j] for k in range(self.K) \
                 for j in range(self.J) for l in range(self.L)) + \
            sum_(cdc[l][j][i]*zdc[l][j][i] for i in range(self.I) \
                 for j in range(self.J) for l in range(self.L))
        )
    
        for l in range(self.L):
            for k in range(self.K):
                sum_(zpd[l][k][j] for j in range(self.J)) <= S[l][k]

        for j in range(self.J):
            sum_(zpd[l][k][j] for k in range(self.K) for l in range(self.L))\
            <= Umax[j]*x[j]

        for l in range(self.L):
            for i in range(self.I):
                sum_(zdc[l][j][i] for j in range(self.J)) == D[l][i]

        for j in range(self.J):
            sum_(zdc[l][j][i] for i in range(self.I) for l in range(self.L))\
            <= Umax[j]*x[j]
                
        for l in range(self.L):
            for k in range(self.K):
                for j in range(self.J):
                    zpd[l][k][j] - Umax[j]*x[j] <= 0

        for l in range(self.L):
            for j in range(self.J):
                for i in range(self.I):
                    zdc[l][j][i] - Umax[j]*x[j] <= 0
        
        for l in range(self.L):
            for j in range(self.J):
                sum_(zpd[l][k][j] for k in range(self.K)) == \
                sum_(zdc[l][j][i] for i in range(self.I))


    def printSolution(self):
        x, zpd, zdc = self.x, self.zpd, self.zdc

        try:
            for l in range(self.L):
                for k in range(self.K):
                    for j in range(self.J):
                        if zpd[l][k][j].val != 0:
                            print('zpd[{}][{}][{}] = {}'.format(l, k, j, zpd[l][k][j].val))
            
            print('----------------------')
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        if zdc[l][j][i].val != 0:
                            print('zdc[{}][{}][{}] = {}'.format(l, j, i, zdc[l][j][i].val))
                            
            self.facilities_operating_cost = 0
            for j in range(self.J):
                self.facilities_operating_cost += self.f[j]*x[j].val
            print('Operating cost for facilities = {:.2f}'\
                  .format(self.facilities_operating_cost))
            
            self.transportation_cost_pd1 = 0
            for l in range(self.L):
                for k in range(self.K):
                    for j in range(self.J):
                        self.transportation_cost_pd1 += self.cpd[l][k][j]*zpd[l][k][j].val
            print('Transportation cost from factories to DC = {:.2f}'\
                  .format(self.transportation_cost_pd1))
            
            self.transportation_cost_dc1 = 0
            for l in range(self.L):
                for j in range(self.J):
                    for i in range(self.I):
                        self.transportation_cost_dc1 += self.cdc[l][j][i]*zdc[l][j][i].val
            print('Transportation cost from DC to customers = {:.2f}'\
                  .format(self.transportation_cost_dc1))
            print('----------------------')
            a = self.operating_cost.index.tolist()
            print('Total cost: {0:.2f}'.format(self.getObjVal()))
            print('Optimal facility locations :')
            for j in range(self.J):
                if x[j].val != 0:
                    print('x[{}] : {}'.format(j, a[j]))
        except:
            print("No feasible soluion found")

    def return_result(self):
        X_list = np.array([])
        for j in range(self.J):
            X_list = np.append(X_list, self.x[j].val)
        try:
            obj_val = self.getObjVal()
        except:
            obj_val = 0
            X_list = np.zeros(self.J)
        return X_list, obj_val
            
    def save_result(self):
        x = self.x
        zpd = self.zpd
        zdc = self.zdc
        a = self.operating_cost.index.tolist() # List of DCs
        b = self.demand.index.tolist() # List of customer locations
        c = self.supply.index.tolist() # List of factories
        
        a_new = [k for k in a if x[a.index(k)].val] # List of active DCs
                    
        result_dict_pd = {}
        writerpd = pd.ExcelWriter('Results/plant_to_DC.xlsx', engine = 'xlsxwriter')
        for l in range(self.L):
            for k in range(len(c)):
                result_list_pd = []
                for j in range(len(a)):
                    if x[j].val != 0:
                        result_list_pd.append(zpd[l][k][j].val)
                result_dict_pd[c[k]] = result_list_pd
            result_df_pd = pd.DataFrame(result_dict_pd, index = a_new).T
            result_df_pd.to_excel(writerpd, sheet_name='product_{}'.format(l+1))
        writerpd.save()
        
        result_dict_dc = {}
        writerdc = pd.ExcelWriter('Results/DC_to_customer.xlsx', engine = 'xlsxwriter')
        for l in range(self.L):
            for i in range(len(b)):
                result_list_dc = []
                for j in range(len(a)):
                    if x[j].val != 0:
                        result_list_dc.append(zdc[l][j][i].val)
                result_dict_dc[b[i]] = result_list_dc
            result_df_dc = pd.DataFrame(result_dict_dc, index = a_new)
            result_df_dc.to_excel(writerdc, sheet_name='product_{}'.format(l+1))
        writerdc.save()

    def plotpie_chart(self):
        fig = plt.figure()
        plt.style.use('seaborn-muted')
#        labels1 = ['CFAs Operating cost', 'SDs Operating cost', 'Transportation cost: \nCFAs'+r'$\rightarrow$'+'SDs',\
#                  'Transportation cost: \nfactories'+r'$\rightarrow$'+'CFAs', 'Transportation cost: \nSDs'+r'$\rightarrow$'+'distributors',\
#                  'Inventory holding cost']
        
        labels2 = ['Facilities \noperating cost \n= {:.2f}'.format(self.facilities_operating_cost), \
                   'Transportation cost: Factories'+r'$\rightarrow$'+'Facilities \n= {:.2f}'.format(self.transportation_cost_pd1),\
                  'Transportation cost: \nFacilities'+r'$\rightarrow$'+'Customers \n= {:.2f}'.format(self.transportation_cost_dc1)]
        
        values = [self.facilities_operating_cost,\
                  self.transportation_cost_pd1,\
                  self.transportation_cost_dc1]
        #explode = (0.1, 0, 0, 0)  # explode 1st slice
        plt.pie(values, 
                labels=labels2, 
                autopct='%1.1f%%', 
                shadow=False, 
                textprops=dict(color='k'),
                startangle=90)
        #plt.legend(labels, loc="upper right",)
        plt.axis('equal')
        #plt.tight_layout()
        fig_title = 'Total cost = {0:.2f}'.format(self.getObjVal())
        fig.suptitle(fig_title, fontsize=12, color = 'k')
        plt.savefig('Results/Pie_chart.png', dpi = 400)
        plt.show()
        
    def plotGraph(self, stra, strb):
        if stra == 'Factory' and strb == 'DC':
            Y = self.zpd
            a = self.supply.index.tolist()
            b = self.operating_cost.index.tolist()
        elif stra == 'DC' and strb == 'Customer':
            Y = self.zdc
            a = self.operating_cost.index.tolist()
            b = self.demand.index.tolist()
        else:
            print('Error in printing')

        a = [stra+' : '+x for x in a]
        b = [strb+' : '+x for x in b]
        
        pos_a, pos_b = {}, {}
        xa = 1.2
        y = 5
        xb = 1.8
        for i in range(len(a)):
            pos_a[a[i]]=[xa,y-i*0.1]
        for i in range(len(b)):
            pos_b[b[i]]=[xb,y-i*0.1]
            
        pos = {}
        pos.update(pos_a)
        pos.update(pos_b)
        
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = [coords[0], coords[1] + 0.022]
        
        for l in range(self.L):
            plt.figure()
            g = nx.Graph()
            g.add_nodes_from(a, bipartite=0)
            g.add_nodes_from(b, bipartite=1)
            nx.draw_networkx_nodes(g,pos_a,nodelist=a,node_color='r',node_size=30,alpha=0.8)
            nx.draw_networkx_nodes(g,pos_b,nodelist=b,node_color='g',node_size=30,alpha=0.8)
            nx.draw_networkx_labels(g,pos_attrs,font_size=7,font_family='sans-serif')
            
            edge_labels = {}
            for j in range(len(a)):
                for i in range(len(b)):
                    if Y[l][j][i].val != 0:
                        g.add_edge(a[j], b[i])
                        edge_labels[(a[j], b[i])] = Y[l][j][i].val
            
            nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g),width=1,alpha=0.8,edge_color='b')
            nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels,font_color='black',\
                                         font_size=6,label_pos=0.5)
            
            plt.xlim((1,2))
            ylimmina = y-0.1*len(pos_a)+0.05
            ylimminb = y-0.1*len(pos_b)+0.05
            ylimmin = min(ylimmina, ylimminb)
            ylimmax = y + 0.05
            plt.ylim((ylimmin,ylimmax))
            plt.axis('off')
            plt.tight_layout()
            #plt.title("Transportation graph for product_{}".format(l+1))
            plt.savefig('Results/{}_to_{}_for_product_{}.png'.format(stra, strb, l+1), dpi = 400)
            plt.show()
       
# Create a directory named 'Results' to save the output results
if not os.path.exists('Results'):
    os.mkdir('Results')
    
demand = pd.read_excel('Data.xlsx', sheet_name='Demand')
supply = pd.read_excel('Data.xlsx', sheet_name='supply')
max_capacity = pd.read_excel('Data.xlsx', sheet_name='Max_capacity')
transportation_cost_pd = pd.read_excel('Data.xlsx', sheet_name='Transportation_cost_pd')
transportation_cost_dc = pd.read_excel('Data.xlsx', sheet_name='Transportation_cost_dc')
operating_cost = pd.read_excel('Data.xlsx', sheet_name='operating_cost')

prob = Facility_Location("Facility Location")
prob.model(demand, supply, max_capacity, transportation_cost_pd, \
           transportation_cost_dc, operating_cost)
prob.optimize()
prob.printSolution()
prob.save_result()
prob.plotGraph('Factory', 'DC')
prob.plotGraph('DC', 'Customer')
prob.plotpie_chart()

#Increase in % demand
dem_percent = np.linspace(0, 100, 40)
objective_value = np.array([])
no_of_DCs = np.array([])
for dp in dem_percent:
    demand_new = demand*(1 + (dp/100))
    prob = Facility_Location("Facility Location")
    prob.model(demand_new, supply, max_capacity, transportation_cost_pd, \
           transportation_cost_dc, operating_cost)
    prob.optimize()
    X_list, obj_val = prob.return_result()
    X_sum = np.sum(X_list)
    objective_value = np.append(objective_value, obj_val)
    no_of_DCs = np.append(no_of_DCs, X_sum)

plt.figure()
plt.plot(dem_percent, objective_value, 'b')
plt.xlabel('% increase in demand')
plt.ylabel('Total cost')
plt.tight_layout()
plt.savefig('Results/Fig_demand_increase_cost.png', dpi = 400)

plt.figure()
plt.plot(dem_percent, no_of_DCs, 'go-')
plt.xlabel('% increase in demand')
plt.ylabel('Number of facilities to be set-up')
plt.tight_layout()
plt.savefig('Results/Fig_demand_increase_no_of_DC.png', dpi = 400)
plt.show()
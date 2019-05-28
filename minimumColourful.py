import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from sys import argv
import argparse
from glob import glob
import copy 


import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from sys import argv
import argparse
from glob import glob
import copy 


def load_args():
	parser = argparse.ArgumentParser(description='Loads path of the data set')
	parser.add_argument('path')
	parser.add_argument('color')
	return parser.parse_args()

def load_dat(path):
	data = None
	with open(path) as f:
		n = int(f.readline().strip()[len('param n := '):-1])
		r = int(f.readline().split(':=')[1])
		data = []
		f.readline()  # ignore line 'param c := '
		for line in f:
			i, j, c_ij = line.strip(';\n').split()
			i, j, c_ij = int(i), int(j), float(c_ij) 
			data.append([i,j,c_ij]) 
	
	return data

def color_assignment(color) : 
	c = []
	with open(color) as f:
		for line in f:
			c.append(line.strip())
	
	return c


class minimum_colourful_subtree:

	""" ############### Methods used for the minimum tree ###############""" 
	def __init__(self,edges,initial_colour_assignment,color_set) :
		self.k = 5
		self.edges = edges
		self.initial_colour_assignment = initial_colour_assignment
		self.color_set = color_set
		self.create_graph() 

	def create_graph(self):
		self.G = nx.DiGraph()
		self.n = len(self.edges)
		self.V = range(self.n)
		
		for i in range(len(initial_colour_assignment)) : 
			self.G.add_node(i,color=self.initial_colour_assignment[i],style='filled',fillcolor=self.initial_colour_assignment[i])

		for i in range(self.n):  
			self.G.add_edge(self.edges[i][0], self.edges[i][1], weight=self.edges[i][2])
		self.draw_graph(self.G)
		
	def draw_graph(self,g) :
		weights = nx.get_edge_attributes(g,'weight')     
		colours = nx.get_node_attributes(g,'color')
		c = list(colours.values()) 
		pos =graphviz_layout(g, prog='dot')
		nx.draw(g, pos, node_color = c,with_labels=True, arrows=True)
		nx.draw_networkx_edge_labels(g,pos,edge_labels=weights)   
		plt.show()

	def get_key(self,item) : 
		return item[2]

	def get_node_num(self, elem) : 
		return self.edges.index(elem)

	def is_colourful(self,g) : 
		""" Check if a tree is colourful """
		colours = nx.get_node_attributes(g,'color')
		c = list(colours.values()) 
		return len(c) == len(set(c))

	def get_graph_colours(self, g) : 
		""" return the list containing the colours of a graph"""
		colours = nx.get_node_attributes(g,'color')
		return list(colours.values()) 

	def get_color_node(self, node) : 
		""" return colour of a node """ 
		colours = nx.get_node_attributes(self.G,'color')
		return colours[node]

	def is_colourful_nodes(self, n1,n2,) : 
		""" check if two nodes are of different colours"""  
		colours = nx.get_node_attributes(self.G,'color')
		return colours[n1] != colours[n2]

	def add_edge_to_graph(self, data, g) :
		n1 = data[0]
		n2 = data[1]
		w = data[2]

		g.add_node(n1,color=self.get_color_node(n1),style='filled',fillcolor=self.get_color_node(n1))  
		g.add_node(n2,color=self.get_color_node(n2),style='filled',fillcolor=self.get_color_node(n2))  

		g.add_edge(n1, n2, weight=w)

	def add_root(self,g) : 
		colours = list(nx.get_node_attributes(self.G,'color').values())
		g.add_node(0,color=colours[0],style='filled',fillcolor=colours[0])

	def find_max_weight(self,source,candidates) : 
		""" return the edge maximising the weight
			first element is the maximum weight 
			second element is the edge endpoint        """ 
		max_w = -1
		max_e = -1
		for i in candidates : 
			if self.G.get_edge_data(source, i)['weight'] > max_w : 
				max_w = self.G.get_edge_data(source, i)['weight']
				max_e = i 

		return[max_w, max_e]

	def get_weight(self,n1,n2) :
		if (self.G.has_edge(n1, n2)): 
			return self.G.get_edge_data(n1, n2)['weight']
		else : 
			return 0

	""" ############### Kruskal style minimum tree ###############""" 
	def kruskal_style(self) :
		#copy original edges 
		kruskal_edges = self.edges[:]
		#sort edges according to their weights in descending order
		kruskal_edges.sort(key=self.get_key, reverse=True)

		#create new empty graph
		self.kruskal_graph = nx.DiGraph()
		
		#add first heaviest weight edge !! 
		heaviest_edge = kruskal_edges[0]
		self.add_edge_to_graph(heaviest_edge, self.kruskal_graph)
		
		temp_kruskal_graph = copy.deepcopy(self.kruskal_graph)

		for i in range(1,len(kruskal_edges)) : 
			self.add_edge_to_graph(kruskal_edges[i],temp_kruskal_graph )
			#self.draw_graph(temp_kruskal_graph)
			
			if (nx.is_tree(temp_kruskal_graph) and self.is_colourful(temp_kruskal_graph) ) : 
				self.add_edge_to_graph(kruskal_edges[i], self.kruskal_graph)
			
			else : 
				temp_kruskal_graph.remove_edge(kruskal_edges[i][0], kruskal_edges[i][1])
				nodes = list(self.kruskal_graph.nodes)
				if kruskal_edges[i][0] not in nodes : 
					temp_kruskal_graph.remove_node(kruskal_edges[i][0])
				if kruskal_edges[i][1] not in nodes : 
					temp_kruskal_graph.remove_node(kruskal_edges[i][1])

		self.draw_graph(self.kruskal_graph)

	""" ############### prim_style minimum tree ###############""" 
	def prim_style(self) : 
		#counter to loop on to have all colours in the graph
		self.counter = self.k 
		#result graph
		self.prim_graph = nx.DiGraph()
		#first add root
		self.add_root(self.prim_graph) 
		#recursivly find on all edges starting with the root
		self.prim_utils(0)
		self.draw_graph(self.prim_graph)
		
	def prim_utils(self, node) : 
		color_list = self.get_graph_colours(self.prim_graph)
		data = []
		children = list(self.G.successors(node))
		candidates = []
		for i in children : 
			c = self.get_color_node(i)
			if c not in color_list : 
				candidates.append(i)
		
		if candidates : 
			data = self.find_max_weight(node,candidates)
		if data : 
			self.add_edge_to_graph([node, data[1], data[0]], self.prim_graph)
			self.prim_utils(data[1])

	""" ############### top_down the minimum tree ###############""" 
	def top_down(self) : 
		#result graph
		self.top_down_graph = nx.DiGraph()
		#first add root
		self.add_root(self.top_down_graph) 
		#temporary test graph 
		self.temp_top_down_graph = copy.deepcopy(self.top_down_graph)

		self.top_down_utils(0)

		self.draw_graph(self.top_down_graph)

	def top_down_utils(self,node) : 
		children = list(self.G.successors(node))
		data = []
		if children : 
			data = self.find_max_weight(node,children)
		if data : 
			self.add_edge_to_graph([node, data[1], data[0]],self.temp_top_down_graph)

			if (nx.is_tree(self.temp_top_down_graph) and self.is_colourful(self.temp_top_down_graph) ) : 
				self.add_edge_to_graph([node, data[1], data[0]], self.top_down_graph)
				self.top_down_utils(data[1])
			
			else : 
				self.temp_top_down_graph.remove_edge(node, data[1])
				nodes = list(self.top_down_graph)
				if node not in nodes : 
					self.temp_top_down_graph.remove_node(node)
				if data[1] not in nodes : 
					self.temp_top_down_graph.remove_node(data[1])

	""" ############### insertion method minimum tree ###############""" 
	def insertion(self) :
		#result graph
		self.insertion_graph = nx.DiGraph()
		#first add root
		self.add_root(self.insertion_graph)
		#colors not in solution yet 
		color_list = self.get_graph_colours(self.insertion_graph)
		temp_color_set = copy.deepcopy(self.color_set)
		temp_color_set.remove(color_list[0])  #to optimize this part 
		
		for c in temp_color_set : 
			#calculate gain
			nodes = list(self.G.nodes)
			output = []
			for v in nodes : 
				if (self.get_color_node(v) == c) :
					output.append(self.calculate_gain(v))
			output.sort(key=self.get_key, reverse=True)
			res = output[0]
					
			if res : 
				# add uv to T 
				data = [res[0], res[1], self.get_weight(res[0],res[1])]
				if data : 
					self.add_edge_to_graph(data, self.insertion_graph)

				# add vx 
				data2 = [res[1], res[3], self.get_weight(res[1], res[3])]
				if data2 :
					self.add_edge_to_graph(data2, self.insertion_graph)
                    
				#remove ux for all x \in T for which w(v,x) > w(u,x)
				for i in self.insertion_graph.nodes : 
					if (self.get_weight(res[1], i) > self.get_weight(res[0],i)) : 
						if self.insertion_graph.has_edge(res[0], i) : 
							self.insertion_graph.remove_edge(res[0], i)
						nodes = list(self.insertion_graph)
						if res[0] not in nodes : 
							self.insertion_graph.remove_node(res[0])
						if i not in nodes : 
							self.insertion_graph.remove_node(i)

		self.draw_graph(self.insertion_graph)
		
	def calculate_gain(self,v) :
		""" Function used by the insertion method to calculate the gain of each vertex """ 
		res = []
		for u in self.insertion_graph.nodes : 
			for x in self.insertion_graph.nodes : 
				w_uv = self.get_weight(u,v) 
				if (self.get_weight(v,x) > self.get_weight(u,x)) : 
					w_uv += self.get_weight(v,x) - self.get_weight(u,x)

				res.append([u,v,w_uv,x])

		res.sort(key=self.get_key, reverse=True)
		return res[0]

	""" ############### critical path method minimum tree ###############""" 
	def critcal_path(self) :
		#result graph
		self.critcal_path_graph = nx.DiGraph()
		#first add root
		self.add_root(self.critcal_path_graph)

		#compute all colourful path from root
		paths = self.find_all_paths(self.G, 0)
		#compute score of each colourful path
		score = self.calculate_score(paths)

		temp_score = copy.deepcopy(score)
		temp_score.sort(reverse=True)
		#max score 
		max = temp_score[0]
		#colouful path with max score
		max_path = paths[score.index(max)]

		#create graph from the maximum colorful path 
		for i in range(len(max_path)-1) : 
			data = [max_path[i], max_path[i+1], self.get_weight(max_path[i],max_path[i+1])]
			self.add_edge_to_graph(data, self.critcal_path_graph)
		self.draw_graph(self.critcal_path_graph)


		

	def calculate_score(self, paths) : 
		score = [0]		
		for i in paths : 
			s = 0 
			if len(i) > 1 : 
				for j in range(len(i)-1) : 
					s += self.get_weight(i[j],i[j+1])
				score.append(s)
		return score	

	def find_all_paths(self, graph, start, path=[], color=[]):
		path = path + [start]
		color = color + [self.get_color_node(start)]

		if start not in graph.nodes:
			return [path]
		paths = [path]
		for node in list(graph.successors(start)):
			if node not in path and self.get_color_node(node) not in color:
				newpaths = self.find_all_paths(graph, node, path, color)
				for newpath in newpaths:
					paths.append(newpath)
	   
		return paths




if __name__ == '__main__':

	edges = load_dat("./dataset/MCS10_1.dat")
	initial_colour_assignment = color_assignment("./dataset/MCS10_1c.dat")	
	print("------------------------initial_colour_assignment--------------------------------------")
	print(initial_colour_assignment)
	c_set = list(set(initial_colour_assignment)) #["red", "blue", "yellow", "green", "orange"]
	initial_graph = minimum_colourful_subtree(edges,initial_colour_assignment,c_set)

	
	initial_graph.kruskal_style()
	#initial_graph.prim_style()

	#initial_graph.top_down()

	#initial_graph.insertion()

	#initial_graph.critcal_path()
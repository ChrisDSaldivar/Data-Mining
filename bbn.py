# Programmer: Chris Saldivar
# coding: utf-8
from collections import deque
from functools import reduce
import copy


class BadQueryError (Exception):
    def __init__ (self, msg):
        super(BadQueryError, self).__init__(msg)

class BBN_Node:
    """Node for a bayesian belief network (must be binary variable)

    """
    def __init__ (self, p_table, label, parents):
        """
        Args:
            p_table (dict): the probability table for the given node. Do not include
                            negation of the node's label as this is calculated in
                            self.get_prob() to save space
            lable (str): the node's label
            parents (list): a list of the node's parents
        """
        self.p_table = p_table
        self.label = label
        self.parents = parents
        
    def __repr__ (self):
        return f'{self.label.upper()}:\n{self.p_table}\nParents:{self.parents}\n'
    
    def get_prob (self, val, parents_vals=''):
        if self.parents == [] and parents_vals != '':
            raise BadQueryError(f"Node {self.label} has no parents")
        
        if self.parents == []:
            prob = self.p_table['t']
            return prob if val == 't' else 1-prob
                
        prob = self.p_table[parents_vals]
        return prob if val == 't' else 1-prob
    
    def has_parents (self):
        return self.parents != []

class BBN:
    """Bayesian Belief Network works for generic network of binary variables
    """
    def __init__ (self, nodes, target, verbose=False):
        """
        Args:
            nodes (list): list of BBN_Nodes pre-initialized
            target (str): the class being estimated
            verbose (bool): flag for verbose output
        """
        self.verbose = verbose
        self.target = target
        self.vars = [node.label for node in nodes if node.label != target]
        self.graph = {}
        for node in nodes:
            self.graph[node.label] = node
            
    def __repr__ (self):
        return str(self.graph)
    
    def _query (self, q):
        probs = []
        for node in self.graph:
            cn = self.graph[node]
            if cn.has_parents():
                p_vals = ''.join([q[p] for p in cn.parents])
                p = cn.get_prob(q[node], p_vals)
                if self.verbose:
                    ps = ','.join([f'{p}={q[p]}' for p in cn.parents])
                    print(f'({node}={q[node]}| {ps}): {p}')
                probs.append(p)
        if self.verbose:
            print()
        return reduce(lambda x,y: x*y, probs)
    
    def classify (self, q):
        """This function will classify the given query

        Accepts full or partial evidence in query. Do not include the target
        variable in the query. This is generated in the method.

        Args:
            q (dict): a dictionary of the evidence provided for classification

        Returns:
            bool: True if class 1 is most probable. False otherwise

        Raises:
            BadQueryError: if the query is malformed
        """
        if q == {}:
            raise BadQueryError('Query is empty')
        for var in q:
            if var not in self.graph:
                raise BadQueryError(f'Variable: "{var}" not in graph')
        unknowns = self._get_unknowns(q)
        if unknowns == []:
            return self._classify_all_evidence(q)
        else:
            return self._classify_partial_evidence(q, unknowns)
        
    def classify_verbose (self, q):
        """
        Same as classify() except it prints the result instead of
        returning it.
        """
        if self.classify(q):
            print(f'{self.target}: True')
        else:
            print(f'{self.target}: False')
        
    def _classify_partial_evidence (self, q, unknowns):
        queries = self._build_queries(q, unknowns)
        target_true_probs = []
        target_false_probs = []
        for query in  queries:
            query[self.target] = 't'
            target_true_probs.append(self._query(query))
            query[self.target] = 'f'
            target_false_probs.append(self._query(query))
        return sum(target_true_probs) > sum(target_false_probs)
    
    def _classify_all_evidence (self, q):
        q[self.target] = 't'
        yes = self._query(q)
        q[self.target] = 'f'
        no = self._query(q)
        if self.verbose:
            print()
            print(f'P(yes):{yes}')
            print(f'P(no):{no}')
        return yes > no
    
    def _get_unknowns (self, query):
        return [node for node in self.graph if node not in query and node != self.target]
        
    def _build_queries (self, q, unknowns):
        queries = deque([q])
        
        for u in unknowns:
            new_stack = []
            for _ in range(len(queries)):
                current_query = queries.popleft()
                tv = copy.deepcopy(current_query)
                tv[u] = 't'
                fv = copy.deepcopy(current_query)
                fv[u] = 'f'
                queries.extend([tv, fv])

        return queries

if __name__ == '__main__':
    # Initialize each node 
    # To change the probability tables simply edit the dict
    e = BBN_Node(
        p_table={'t': .7}, 
        label='e',
        parents=[]
    )
    d = BBN_Node(
        p_table={'t': .25}, 
        label='d',
        parents=[]
    )
    hd = BBN_Node(
        p_table={'tt': .25, 'tf': .45, 'ft': .55, 'ff': .75}, 
        label='hd',
        parents=['e', 'd']
    )
    cp = BBN_Node(
        p_table={'t': .8, 'f': .01}, 
        label='cp',
        parents=['hd']
    )
    bp = BBN_Node(
        p_table={'t': .85, 'f': .2}, 
        label='bp',
        parents=['hd']
    )

    # Create Bayesian Belief Network object and set target variable for classify
    bbn = BBN(nodes=[e, d, hd, cp, bp], target='hd', verbose=True)


    # For the query make a dictionary like:
    #  {node_label: truth_value,
    #   node_label: truth_value
    #  }
    # node_label != target variable and must match one of the labels given to a node
    query = {'e': 'f', 'd': 't', 'cp': 't', 'bp': 't' } # Example from slides

    # Classify the given query
    bbn.classify_verbose(query)
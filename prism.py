# Chris Saldivar

# coding: utf-8

import pandas as pd
from operator import and_
from functools import reduce
from collections import Counter
import sys

# options set for jupyter notebook
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class Prism:
	def __init__ (self, file_path, header=None):
		self.read_data_csv(file_path, header)

	def read_data_csv (self, file_path, header=None):
		try:
			self.df = pd.read_csv(file_path, header=header)
		except FileNotFoundError as e:
			print(f'ERROR: File {file_path} not found')
			sys.exit(1)
		self.rules = []
		if header is None: # make column names [A...Z-A1...Z1-A2...Z2-A3...Z3 etc]
			self.df.columns = [chr(i%26+65) + (str((i//26) % 26) if i//26 != 0 else '') for i in range(len(self.df.columns))]
		# Get the least common class
		self.class_column = self.df.columns[-1]
		self.attributes = list(self.df.columns[:-1])
		self.classes = Counter(self.df[self.class_column])
		self.classes = self.classes.most_common()[:-len(self.classes)-1:-1]

	def __drop_rows (self, rule):
		indices = self.df[reduce(and_, [self.df[attr] == value for attr, value in rule['lhs'].items()])].index
		self.df.drop(indices, inplace=True)

	def __gen_slice (self, rule):
		if rule == {}:
			return self.df
		else:
			return self.df[reduce(and_, [self.df[attr] == value for attr, value in rule['lhs'].items()])]

	def __eps_equals (self, lhs, rhs, epsilon=1e-3):
		return abs(lhs - rhs) < epsilon 

	def __increment_counts (self, counts, row, attr, curr_class):
		if counts[attr].get(row[attr]) is None:              # if this is a new (attr, value) pair
			counts[attr][row[attr]] = {'tot': 1, 'cnt': 0}   # init its counts
		else:
			counts[attr][row[attr]]['tot'] += 1          # otherwise increment its total count
		if row[self.class_column] == curr_class:              # if this is a potential rule covering the class
			counts[attr][row[attr]]['cnt'] += 1          # count it

	def __choose_test (self, counts):
		max_accuracy = -1
		max_coverage = -1
		chosen_test = ()
		for attr in counts:
			for value in counts[attr]:
				p = counts[attr][value]['cnt']
				t = counts[attr][value]['tot']
				accuracy = p/t
				if ((self.__eps_equals(accuracy, max_accuracy) and t > max_coverage)
				   or accuracy > max_accuracy):
					max_coverage = t
					max_accuracy = accuracy
					chosen_test = (attr, value, max_accuracy)
		return chosen_test

	def __display_rule (self, rule):
		print('if', end=' ')
		for i, attr in enumerate(rule['lhs']):
			print(f'{attr}={rule["lhs"][attr]}', end=' ')
			if i < len(rule['lhs'])-1:
				print('and', end=' ')
		print(f'then {rule["rhs"][0]}={rule["rhs"][1]}')

	def generate_rules (self):
		for curr_class, _ in self.classes:
			# This is used for slice the dataframe based on the rule
			while curr_class in self.df[self.class_column].values:     # generate rules for the current class
				curr_df = self.df.copy()
				curr_attr_list = self.attributes.copy()    # used so we only look at attributes no in the rule
				rule = {'lhs': {},                      # initialize the rule with empty lhs and curr_class
						'rhs': (self.class_column, curr_class)
				}
				accuracy = 0.0                         # accuracy set to 0 TODO: calc this
				while not(self.__eps_equals(accuracy, 1.0)) and curr_attr_list:  # until the rule is perfect
					counts = {}
					for attr in curr_attr_list:                           # count the values for conditions
						counts[attr] = {}
						for idx, row in curr_df.iterrows():
							self.__increment_counts(counts, row, attr, curr_class)
					chosen_attr, chosen_val, accuracy = self.__choose_test(counts)
					curr_attr_list.remove(chosen_attr)
					rule['lhs'][chosen_attr] = chosen_val
					curr_df = self.__gen_slice(rule)
				self.__drop_rows(rule)

				self.rules.append(rule)

	def display_rules (self):
		print('\nGenerated Rules\n'+'-'*80)
		for rule in self.rules:
			self.__display_rule(rule)



if __name__ == '__main__':
	file_path = input('Please enter the csv filename (with .csv extensions): ')
	header = input('Is this file\'s first line headers? (y/n): ').lower()
	header = 0 if len(header) > 0 and header[0] == 'y' else None

	prism = Prism(file_path, header)
	prism.generate_rules()
	prism.display_rules()
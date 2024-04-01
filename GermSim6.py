#!/home/zachary/miniconda3/envs/netmotifs_py3/bin/python3
# Put a shebang line here to run as a python script automatically.

"""Simple evolution simulator designed to test the evolutionary favorability of altruism."""

from random import choice, shuffle, randrange, random, seed
from functools import reduce, partial
import time
from matplotlib import pyplot as plt

import numpy as np

seed('jwdowfourwfow')

hos = 0.55  # base odds of survival
mr = 10  # range within which mating may occur
hr = 5  # range within which helping may occur
r = 2  # number of babies each germ can make
mig = 10  # maximum migration distance
give = 0.25  # helping germs lose this much fitness
take = 0.33  # helped germs are this much more fit
friends = 10  # number of germs within ir before half of its fitness is lost.
# Set this value zero to negate competition.
region = (500, 500)  # germs start in a 500X500 area
ir = 2  # inhibition radius

altvals = {'AA': (True, True), 'Aa': (True, False), 'aa': (False, False)}

# divide an area nine times the starting region into boxes for easy computation later.
mdi = max(mr, hr, ir)  # Maximum distance of interaction
xboxes = int(3 * region[0]/mdi) + 2
yboxes = int(3 * region[1]/mdi) + 2
# Make a dictionary of lists. The lists will collectively contain each germ.
# The keys of lit_boxes will be max x- and y-coordinates in a tuple.
lit_boxes = {}
big_boxes = {}
for x in range(xboxes):
	for y in range(yboxes):
		lit_boxes[(x * mdi, y * mdi)] = []
		big_boxes[(x * mdi, y * mdi)] = []

k = 5.288267030694529/friends


def sigmoid(num):
	"""Sigmoid function of x"""
	if num < 10000 + friends:
		return 1 / (1 + 2 ** (k*num - k*friends))
	else:
		return 0


# Make a dictionary of likely numbers of neighbors in inhibition range
sigvals = {}
for key in range(51):
	sigvals[key] = sigmoid(key)


# Dictionary for migratory translations
mig_trans = {0: (0, 5), 1: (0, 6), 2: (0, 7), 3: (0, 8), 4: (0, 9), 5: (0, 10),
			 6: (2, 5), 7: (2, 6), 8: (3, 6), 9: (3, 7), 10: (3, 8), 11: (4, 9),
			 12: (4, 4), 13: (4, 4), 14: (5, 5), 15: (6, 6), 16: (6, 6),
			 17: (7, 7), 18: (5, 2), 19: (6, 2), 20: (6, 3), 21: (7, 3),
			 22: (8, 3), 23: (9, 4), 24: (5, 0), 25: (6, 0), 26: (7, 0),
			 27: (8, 0), 28: (9, 0), 29: (10, 0), 30: (5, -2), 31: (6, -2),
			 32: (6, -3), 33: (7, -3), 34: (8, -3), 35: (9, -4), 36: (4, -4),
			 37: (4, -4), 38: (5, -5), 39: (6, -6), 40: (6, -6), 41: (7, -7),
			 42: (2, -5), 43: (2, -6), 44: (3, -6), 45: (3, -7), 46: (3, -8),
			 47: (4, -9), 48: (0, -5), 49: (0, -6), 50: (0, -7), 51: (0, -8),
			 52: (0, -9), 53: (0, -10), 54: (-2, -5), 55: (-2, -6),
			 56: (-3, -6), 57: (-3, -7), 58: (-3, -8), 59: (-4, -9),
			 60: (-4, -4), 61: (-4, -4), 62: (-5, -5), 63: (-6, -6),
			 64: (-6, -6), 65: (-7, -7), 66: (-5, -2), 67: (-6, -2),
			 68: (-6, -3), 69: (-7, -3), 70: (-8, -3), 71: (-9, -4),
			 72: (-5, 0), 73: (-6, 0), 74: (-7, 0), 75: (-8, 0), 76: (-9, 0),
			 77: (-10, 0), 78: (-5, 2), 79: (-6, 2), 80: (-6, 3), 81: (-7, 3),
			 82: (-8, 3), 83: (-9, 4), 84: (-4, 4), 85: (-4, 4), 86: (-5, 5),
			 87: (-6, 6), 88: (-6, 6), 89: (-7, 7), 90: (-2, 5), 91: (-2, 6),
			 92: (-3, 6), 93: (-3, 7), 94: (-3, 8), 95: (-4, 9)}


class Germ:
	__slots__ = ('FIT', 'comp', 'helped', 'helping', 'altruism', 'location',
				 'box')

	def __init__(self, al=None, p1=None, p2=None):
		self.FIT = hos  # germs' base chance of survival
		self.comp = 0
		self.helped = False  # germs are not born helped
		self.helping = False  # germs are not born helping
		if p1 is None or p2 is None:
			"""If the germ has no parents, it gets a random location and
			assigned altruism value."""
			self.altruism = altvals[al]
			self.location = (randrange(0, region[0], 10) + region[0],
							 randrange(0, region[1], 10) + region[1])
		else:
			"""If the germ has parents, it starts in between them and migrates.
			It inherits a random altruism allele from each parent"""
			self.altruism = (choice(p1.altruism), choice(p2.altruism))
			loc = np.round(np.array((p1.location, p2.location)).mean(axis=0))
			migration = mig_trans[randrange(96)]
			self.location = tuple(np.array((loc, migration)).sum(axis=0))
		x = self.location[0]
		y = self.location[1]
		x_marker = int(x / mdi) * mdi
		y_marker = int(y / mdi) * mdi
		self.box = (x_marker, y_marker)
		return

	def help(self, partner):
		"""Germs improve other germs' fitness by reducing their own."""
		self.FIT = self.FIT - give
		self.helping = True
		partner.FIT += take
		partner.helped = True
		return


def catalogue(pop1):
	"""Put all the germs in the population into boxes"""
	# Get rid of the old references
	for d in lit_boxes:
		lit_boxes[d] = []
	for d in big_boxes:
		big_boxes[d] = []

	# For each germ, put it in a little box and nine big boxes corresponding to
	# its coordinates.
	for germ in pop1:
		x, y = germ.box
		# Each germ gets one little box and nine big boxes.
		lit_boxes[(x, y)].append(germ)
		for a in range(x - mdi, x + 2*mdi, mdi):
			for b in range(y - mdi, y + 2*mdi, mdi):
				big_boxes[(a, b)].append(germ)
	return 0


def compete(germ):
	""""Germs compete for resources, reducing fitness of neighbors
	incrementally."""
	if friends == 0:  # Exception for Utopia.
		return
	for neighbor in big_boxes[germ.box]:
		# if np.sum(np.subtract(germ.location, neighbor.location) ** 2
		#		   <= ir ** 2):
		if (germ.location[0] - neighbor.location[0])**2 +\
		   (germ.location[1] - neighbor.location[1])**2 <= ir**2:
			if germ is not neighbor:
				neighbor.comp += 1
	return


def decide(germ1, pop1):
	"""Each germ decides if it wants to help a random nearby germ. Germs with
	dominant alleles will always help if able; germs with the aa genotype will
	never help."""
	if germ1.altruism[0] or germ1.altruism[1]:  # If the germ is altruistic
		for germ2 in big_boxes[germ1.box]:
			# Using sum squared of the difference in x and y positions to find
			# distance squared:
			if (germ1.location[0] - germ2.location[0])**2 +\
			   (germ1.location[1] - germ2.location[1])**2 <= hr**2:  # if germs are within helping range
				if germ1 is not germ2 and germ2.helped is False:
					germ1.help(germ2)
					break
	return

try:
	#import graphics as gp  # pip3 install --user http://bit.ly/csc161graphics

	def display(pop1):
		"""Display the location and genotype of each germ in the meadow."""
		start_time = time.time()
		loco = []
		X = []
		Y = []
		C = []
		for germ in pop1:
			x = germ.location[0] / 2
			X.append(x)
			y = germ.location[1] / 2
			Y.append(y)
			al = germ.altruism
			if al == (True, True):
				color = 'blue'
			elif al == (True, False) or al == (False, True):
				color = 'purple'
			else:
				color = 'red'
			C.append(color)
			loco.append((x, y, color))
		#win = gp.GraphWin("Germ_Meadow", 750, 750)
		fig, ax = plt.subplots()
		ax.scatter(X, Y, c=C)
		#for dot in loco:
		#	pt = gp.Point(dot[0], dot[1])
		#	pt.draw(win)
		#	pt.setFill(dot[2])
		plt.show(block=True)
		print("\tdone in %.2f seconds" % (time.time() - start_time))
		#input()
		plt.close(fig)
		return

except:
	print("Warning: This program could not find the graphics module. Nothing can"
			  " be displayed.")

	def display(pop1):
		"""Do nothing because the graphics module isn't installed."""
		return


def found(hd, hetero, hr):
	"""for each possible genotype, put a number of germs randomly in the
	starting region with that genotype"""
	homo_dom = [Germ('AA') for i in range(hd)]
	het = [Germ('Aa') for i in range(hetero)]
	homo_rec = [Germ('aa') for i in range(hr)]
	prov = homo_dom + het + homo_rec
	return prov


def getResult(prov):
	"""Returns the number of germs of each genotype."""
	result = {'AA': 0, 'Aa': 0, 'aa': 0}
	for germ in prov:
		if germ.altruism == (True, True):
			result['AA'] += 1
		elif germ.altruism == (True, False) or germ.altruism == (False, True):
			result['Aa'] += 1
		elif germ.altruism == (False, False):
			result['aa'] += 1
		else:
			print("there was a problem getting results")
	return result


def mate(germ1):
	"""Each germ tries to find a partner that isn't itself within MR units.
	They make 2r children."""
	babies = []
	for germ2 in big_boxes[germ1.box]:
		# using sum squared of the difference in x and y positions to find
		# distance
		if np.sum(np.subtract(germ1.location, germ2.location) ** 2) <= mr ** 2:
			if germ1 is not germ2:
				babies = [Germ(p1=germ1, p2=germ2) for i in range(r)]
				break
	return babies


def select(pop1):
	"""Kill germs that don't pass this fitness test. Being helped improves
	chances, while helping reduces chances"""
	prov2 = list()
	for germ in pop1:
		rando = random()
		comp = germ.comp
		if rando < germ.FIT * sigvals.get(comp, sigmoid(comp)):
			prov2.append(germ)
	return prov2


def cycles(report, trials, gens, homodom, hetero, homorec, seed_file=None):
	report_file = open(report, 'w')

	# write the file header
	report_file.write("%d trials\n%d generations per trial\n"
					  "AA: %d; Aa: %d, aa: %d\n"
					  "Base odds of survival: %f\n"
					  "Mating maximum radius: %d\n"
					  "Helping maximum radius: %d\n"
					  "Number of offspring per mating pair: %d\n"
					  "Maximum migration distance: %d\n"
					  "Fitness lost when Helping: %f\n"
					  "Fitness gained when helped: %f\n"
					  "Competition parameter: %f\n"
					  "Radius of competitive inhibition: %d\n"
					  "Starting region: %d X %d\n"
					  % (trials, gens, homodom, hetero, homorec, hos, mr, hr,
						 r, mig, give, take, friends, ir, region[0], region[1]))

	# Prepare seeds
	if seed_file is not None:
		seed_list = [line for line in open(seed_file, 'r')]
		if len(seed_list) <= trials:
			seed_list = None
	else:
		seed_list = None

	totals = {"AA": 0, "Aa": 0, "aa": 0}

	# Do the trials.
	for trial in range(trials):
		start_time = time.time()
		print("simulating trial %d." % trial)
		if seed_list is None:
			seed()
		else:
			seed(seed_list[trial])
		population = found(homodom, hetero, homorec)
		for generation in range(gens):
			catalogue(population)
			list(map(compete, population))
			shuffle(population)
			list(map(partial(decide, pop1=population), population))
			population = select(population)
			shuffle(population)
			population2 = []
			for germ in population:
				population2.extend(mate(germ))
			population = population2
		compo = getResult(population)
		totals["AA"] += compo["AA"]
		totals["Aa"] += compo["Aa"]
		totals["aa"] += compo["aa"]
		report_file.write("\nTrial %d:\tAA:%d\tAa:%d\t"
						  "aa:%d\t Total:%d\n"
						  % (trial, compo['AA'], compo['Aa'], compo['aa'],
							 compo['AA'] + compo['Aa'] + compo['aa']))
		print("Done in %.2f seconds." % (time.time() - start_time))

	# Write the file footer.
	report_file.write("\nTotal AA germs:%d\nTotal Aa germs:%d"
					  "\nTotal aa germs:%d\nMean AA germs:%.4f\n"
					  "Mean Aa germs:%.4f\nMean aa germs:%.4f\n"
					  % (totals["AA"], totals["Aa"], totals["aa"],
						 float(totals["AA"])/trials, float(totals["Aa"]) /
						 trials, float(totals["aa"])/trials))
	return


def main():
	founders = {'AA': 0, 'Aa': 0, 'aa': 0}
	for genotype in founders:
		while True:
			print("Please enter the number of " + genotype + " individuals")
			trial = input()
			if trial.isdigit():
				break
			else:
				print("Integer input is required. Please try again.")
		founders[genotype] = int(trial)

	population = found(founders['AA'], founders['Aa'], founders['aa'])
	display(population)

	while True:
		trial = input("How many generations would you like to test?")
		if (not trial.isdigit()) and trial != "indefinite":
			print("Integer or \"Indefinite\" input is required. Please try"
				  "again.")
		elif trial.isdigit():
			gens = int(trial)
			break
		elif trial == "indefinite":
			gens = 0
			break
	generation = 0
	while True:
		print("cataloguing germs")
		start_t = time.time()
		catalogue(population)
		print("\tdone in %.2f seconds" % (time.time() - start_t))
		print("adjusting for competition")
		start_t = time.time()
		list(map(compete, population))
		print("\tdone in %.2f seconds" % (time.time() - start_t))
		print("deciding which germs will help")
		start_t = time.time()
		shuffle(population)
		list(map(partial(decide, pop1=population), population))
		print("\tdone in %.2f seconds" % (time.time() - start_t))
		print("selecting survivors")
		start_time = time.time()
		population = select(population)
		print("\tdone in %2f seconds" % (time.time() - start_time))
		print("creating the next generation")
		start_time = time.time()
		population2 = []
		for germ in population:
			population2.extend(mate(germ))
		population = population2
		print("\tdone in %.2f" % (time.time() - start_time))
		print("generation %d :" % generation)
		print(getResult(population))
		print("displaying population")
		display(population)
		generation += 1
		if trial.isdigit():
			if generation >= gens:
				break
		else:
			request = input("type quit to quit, anything else to "
							"continue.")
			if request == 'quit':
				break


if __name__ == "__main__":
	main()

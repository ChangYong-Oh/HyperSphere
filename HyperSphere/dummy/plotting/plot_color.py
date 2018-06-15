

def algorithm_color(algorithm):
	if algorithm == 'hyperopt':
		return 'r'
	if algorithm == 'smac':
		return 'dodgerblue'
	if algorithm == 'spearmint':
		return 'darkorchid'
	if algorithm == 'spearmint_warping':
		return 'indigo'
	if algorithm == 'cube':
		return 'salmon'
	if algorithm == 'cubeard':
		return 'r'
	if algorithm[:10] == 'additiveBO':
		return 'g'
	if algorithm == 'elasticGP':
		return 'darkslategray'
	if algorithm == 'sphereboth':
		return 'green'
	if algorithm == 'sphereorigin':
		return 'limegreen'
	if algorithm == 'spherewarpingboth':
		return 'lime'
	if algorithm == 'spherewarpingorigin':
		return 'lime'
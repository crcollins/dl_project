import os
import sys

import numpy

from molml.features import EncodedBond, BagOfBonds, LocalEncodedBond, Shell


def load_xyz(path):
	elements = []
	coords = []
	try:
		with open(path, "r") as f:
			for i, line in enumerate(f):
				if i < 2:
					continue
				line = line.strip()
				ele, x, y, z = line.split()
				elements.append(ele)
				coords.append((float(x), float(y), float(z)))
	except IOError:
		return None
	return elements, numpy.array(coords)


if __name__ == '__main__':
	in_base = sys.argv[1]
	out_base = sys.argv[2]
	name = in_base.split("_")[0]
	files = os.listdir(in_base)
	max_name = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
	parts = max_name[:-4].split('_')
	max_num = int(parts[-1])
	name = '_'.join(parts[:-1])
	string = os.path.join(in_base, "%s_%s.xyz" % (name, "%06d"))
	
	elements, coords = load_xyz(os.path.join(in_base, max_name))
	#trans = EncodedBond(input_type=("elements", "coords"))#, max_depth=3)#segments=200)
	trans = LocalEncodedBond(input_type=("elements", "coords"))#, max_depth=3)#segments=200)
	trans.fit([(elements, coords)])

	max_num = 50*1000+2
	eps = 1e-3	
	for i in xrange(1, max_num, 1000):
		outname = os.path.join(out_base, "%06d.npy" % i)
		if os.path.exists(outname):
			continue
		print "Making %s" % outname
		# create a "lock"
		with open(outname, 'w') as f:
			pass
		
		all_data = [load_xyz(string % j) for j in xrange(i, i + 1000)]
		all_data = [x for x in all_data if x is not None]

		axes = [
			numpy.array([eps,  0.,  0.]),
			numpy.array([ 0., eps,  0.]),
			numpy.array([ 0.,  0., eps]),
		]
		all_feats = []
		for elements, coords in all_data:
			data = []
			for i, x in enumerate(coords):
				for axis in axes:
					forward = coords.copy()
					backward = coords.copy()
					forward[i,:] += axis
					backward[i,:] -= axis
					data.append((elements, forward))
					data.append((elements, backward))

			feats = trans.transform(data)
			all_feats.append(feats)
		numpy.save(outname, numpy.array(all_feats))

import csv
import sys

#Filename,Name,Type,ExactName,Features,Options,HOMO,LUMO,HomoOrbital,Dipole,Energy,BandGap,Time,DipoleVector,ExcitationDipoleVector,OscillatorStrength,SpatialExtent,StepNumber

with open(sys.argv[1]) as f:
	reader = csv.reader(f)
	for i, row in enumerate(reader):
		if not i: continue
		print ' '.join(row[6:8] + row[9:11] + [row[-2]])

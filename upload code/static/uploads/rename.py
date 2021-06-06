import glob
import os
crack = 1
wheeze = 1
normal = 1
for file in glob.glob("*.wav"):
	if(file[0] == 'c'):
		num = crack
		crack += 1
	elif(file[0] == 'n'):
		num = normal
		normal += 1
	else:
		num = wheeze
		wheeze += 1
	new_name = file[0:6]+str(num)+file[len(file)-4:]
	print(new_name)

	os.rename(file,new_name)

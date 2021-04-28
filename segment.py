# python code to segment the audido signal into wheeze crack and normal andcopy to folders
import glob
import os
from pydub import AudioSegment
dir = os.getcwd()
os.chdir(dir)
crack = 0
wheeze = 0
normal = 0
count = 0
sample_count = 0
for file in glob.glob("*.txt"):  # select all text file
    print(file)
    print(sample_count)
    sample_count += 1
    length = len(file)
    filename = file[:length-4]
    aud_filename = filename+".wav"  # audio file name is saved
    split = filename.split("_")
    channel = split[3]
    if(channel == "mc"):
        continue
    print(aud_filename)
    try:
        f = open(file)
        segment_no = 0
        for line in f:
            # print(line) ,
            data = line.split()
            segment_no += 1
            # print(data[0])
            t1 = float(data[0])
            t2 = float(data[1])
            t1 = t1 * 1000  # Works in milliseconds
            t2 = t2 * 1000
            newAudio = AudioSegment.from_wav(aud_filename)
            newAudio = newAudio[t1:t2]
            count += 1
            #new_name = 'part' + str(count) + '.wav'
            # Exports to a wav file in the current path.
            #newAudio.export(new_name, format="wav")
            new_name = '_' + filename + '_s' + str(segment_no) + '_' + '.wav'
            if(data[2] == '0' and data[3] == '0'):
                normal += 1
                new_name = 'normal_' + str(normal) + '_' + new_name
                newAudio.export('Normal/'+new_name, format="wav")
            elif(data[2] == '1'):
                crack += 1
                new_name = 'crack_' + str(crack) + '_' + new_name
                newAudio.export('Crack/'+new_name, format="wav")
            else:
                wheeze += 1
                new_name = 'wheeze_' + str(wheeze) + '_' + new_name
                newAudio.export('Wheeze/'+new_name, format="wav")
    finally:
        f.close()
